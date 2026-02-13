"""
GW response computation functions. This module is meant to replace `fast_geometry` for
fixed-skymap pixel responses.

The main procedure is `calculate_response_functions`, which is used to interface with
the rest of BLIP.

The objective is to compute the LISA response for a given sky direction, frequency and
time. This is done in `mich_response_unconvolved`.
"""

# This module leans heavily on JAX automatic vectorization and JIT compilation. All the
# functions are written for the simplest possible array shapes (mostly scalars), making
# them easier to check for correctness. Trace-time assertions, mostly on array shapes,
# have also been placed as strong comments all throughout the code.

# The formulas implemented here are exactly the ones in the original BLIP paper
# Banagiri+21, with two exceptions:
# - the mistaken factor of 1/4pi is removed from eq. (18) which defines the response
#   matrix;
# - all throughout this module, the sign of n is flipped wrt. the paper, i.e. the vector
#   n here means the direction towards the GW source, not from the source.

# TODO check if the sign of n conflicts with the rest of BLIP.

import functools
import logging

import jax
from jax import numpy as jnp, vmap, jit
import chex
import numpy as np
from tqdm import tqdm

import healpy as hp
from astropy import units
from astropy.coordinates import SkyCoord
from scipy.special import sph_harm_y

import lisaorbits
from lisaconstants import SPEED_OF_LIGHT as CLIGHT, ASTRONOMICAL_YEAR as YEAR

from . import submodel

ARMLENGTH = 2.5e9
FSTAR = CLIGHT / (2 * jnp.pi * ARMLENGTH)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())


# This is the only procedure where plain numpy is used.
def calculate_response_functions(freqs, times, submodels, params, plot_flag=False):
    """
    Compute SGWB responses for each submodel and write it to the submodel attributes.

    This procedure avoids redundant calculations by only computing the response for a
    given pixel once. It also uses a sparse time-frequency grid and linearly
    interpolates the results in-between.

    It mirrors the functionality in fast_geometry.calculate_response_functions().

    Parameters
    ----------
    freqs : array (nfreqs,)
        frequencies.
    times : array (ntimes,)
        times.
    submodels : list[submodel]
        The submodels whose response matrices will be computed. They must be in pixel
        basis and have a skymap attribute, not assumed normalized.

        These objects also act as output parameters: after the procedure has run, each
        of them will receive a response matrix as an attribute, which is an array of
        shape (3, 3, nfreqs, ntimes).

        The attributes used for the output depend on the value of plot_flag: the
        response matrix will be written to sm.response_mat and sm.inj_response_mat if it
        is False, or to sm.fdata_response_mat if it is True.
    params: dict
        Parameter dictionary from parse_config().
    plot_flag : bool, optional
        If True, don't write the output response matrix to sm.response_mat or
        sm.inj_response_mat, but to submodel.fdata_response_mat. This is useful for
        plotting simulation and analysis models together (hence the name). Defaults to
        False.

    Warnings
    --------
    This is experimental, tested only for healpix responses.
    """

    chex.assert_rank([freqs, times], 1)
    for sm in submodels:
        assert isinstance(sm, submodel.submodel)
        assert hasattr(sm, "has_map")
        assert sm.has_map or sm.fullsky

    orbits = compute_orbits(times)

    npix = hp.nside2npix(params["nside"])
    is_fullsky = any([getattr(sm, "fullsky", None) for sm in submodels])
    if is_fullsky:
        active_pixels_idx = jnp.arange(npix)
    else:
        active_pixels_idx_sm = [
            _intarray_to_set(jnp.nonzero(sm.skymap)[0])
            for sm in submodels
            if sm.has_map
        ]
        active_pixels_idx = jnp.array(
            sorted(
                list(functools.reduce(lambda x, y: x.union(y), active_pixels_idx_sm))
            ),
            dtype=int,
        )

    assert active_pixels_idx.dtype == int

    active_pixels_vecs = all_unit_vecs_healpix(params["nside"])[active_pixels_idx]
    chex.assert_shape(active_pixels_vecs, (None, 3))

    # Vectorize twice: on times and on frequencies.
    # Sky directions are done sequentially
    ## this vectorizes on times
    _mru_vect = vmap(
        mich_response_unconvolved, (0, None, None, None), out_axes=2
    )  # (3, 3, ntimes)
    ## this vectorizes on frequencies
    _mru_vec2 = vmap(
        _mru_vect, (None, 0, None, None), out_axes=2
    )  # (3, 3, nfreqs, ntimes)
    ## jit the time x frequency calculation
    mru_vec2 = jit(_mru_vec2)

    ## get the sparse time-frequency grid to interpolate on later
    times_sparse, freqs_sparse = get_sparse_tf_grid(times, freqs)
    nt, nf, nt_s, nf_s = len(times), len(freqs), len(times_sparse), len(freqs_sparse)

    ## check runtime, memory use
    _compiled = mru_vec2.lower(
        times_sparse, freqs_sparse, active_pixels_vecs[0], orbits
    ).compile()
    logger.debug("freq range %f - %f", freqs[0], freqs[-1])
    logger.debug("sparse ntimes = %d, sparse nfreqs = %d", nt_s, nf_s)
    logger.debug("response execution cost analysis: %s", _compiled.cost_analysis())
    logger.debug("response memory cost analysis: %s", _compiled.memory_analysis())
    # From a CPU test:
    #    output size =  144 bytes / time / freq / pixel (one complex 3x3 matrix)
    # bytes accessed = 3049 bytes / time / freq / pixel
    #          flops = 1378  flop / time / freq / pixel

    # NOTE the usage of 3x3 complex matrices in our context is a waste of RAM. We assume
    # equal and constant LTTs, so our SGWB can be perfectly described by two real
    # series: the AA and EE time-varying PSDs. That would reduce the output from 144
    # bytes to 16 bytes per time per frequency per pixel (9x compression). And that's
    # still using double precision, which we don't need for the response matrix output.
    # Using single precision we could do 144 -> 8 bytes, an 18x compression.
    ## However, the matrix inversions in the likelihood require double precision,
    # and the spherical harmonic search requires the complex responses.

    ## Loop over sky directions sequentially, computing the per-pixel responses as we go
    ## We only do this on pixels with power
    print("Computing LISA response functions...")
    responses_sparse = []
    for i, pix_vec in tqdm(
        zip(active_pixels_idx, active_pixels_vecs),
        total=len(active_pixels_idx),
        desc="response",
        unit="pixel",
    ):
        responses_sparse.append(mru_vec2(times_sparse, freqs_sparse, pix_vec, orbits))

    ## these should always be 3 x 3 x frequencies x times
    chex.assert_shape(responses_sparse[0], (3, 3, nf_s, nt_s))
    chex.assert_equal_shape(responses_sparse)

    # Integrate on sky for each submodel. We do this sequentially with a python for loop
    # and using plain numpy (not jax) arrays. The point of this is to avoid allocating
    # memory for the whole integrand, a large rank-5 tensor.

    ## get dOmega
    dOmega = 4*np.pi / npix

    ## set up the interpolator
    _interpolate = get_response_interpolator(times, freqs, times_sparse, freqs_sparse)
    _interpolate = jit(_interpolate)
    for sm in submodels:
        ## models with maps
        if sm.has_map:
            ## models with fixed templates
            if not sm.parameterized_map:
                if sm.basis == "pixel":
                    ## check that the skymap has the correct number of pixels
                    chex.assert_shape(sm.skymap, (npix,))

                    ## 3 x 3 x frequency x time
                    integral = np.zeros((3, 3, nf, nt))
                    postf_dims = 1  ## just time

                    ## normalize skymap such that it integrates to 1 over the sky
                    skymap_normalized = sm.skymap / (jnp.sum(sm.skymap) * dOmega)

                    ## loop over pixels, interpolating the response as we go
                    ## only loop over pixels in the skymap with nonzero power
                    for i, response_sparse in zip(active_pixels_idx, responses_sparse):
                        response = _interpolate(response_sparse)
                        integral += skymap_normalized[i] * response * dOmega

                elif sm.basis == "sph":
                    alm_size = (sm.almax + 1) ** 2
                    ## angular coordinates of pixel indices
                    theta, phi = hp.pix2ang(params["nside"], active_pixels_idx)
                    Ylms = np.zeros((npix, alm_size), dtype="complex")
                    ## Get the spherical harmonics
                    for ii in range(alm_size):
                        lval, mval = sm.idxtoalm(sm.almax, ii)
                        Ylms[:, ii] = sph_harm_y(
                            mval, lval, theta, phi
                        )  ## theta, phi switched due to new scipy convention
                    ## check that the Ylms have the right number of pixels, sph terms
                    chex.assert_shape(Ylms, (npix, alm_size))

                    ## 3 x 3 x frequency x time x Ylms
                    integral = np.zeros((3, 3, nf, nt), dtype="complex")
                    postf_dims = 1  ## time x Ylms

                    ## loop over pixels, interpolating the response as we go
                    ## only loop over pixels in the skymap with nonzero power
                    for i, response_sparse in zip(active_pixels_idx, responses_sparse):
                        response = _interpolate(response_sparse)
                        pix_sph_sum = np.sum(
                            Ylms[i, :] * sm.alms_inj
                        )  ## sum over Ylms for this pixel
                        integral += pix_sph_sum * response * dOmega

                else:
                    assert False

            ## parameterized spatial models
            else:
                if sm.basis == "pixel":

                    ## 3 x 3 x frequency x time
                    sky_size = np.sum(sm.mask_idx)
                    integral = np.zeros((3, 3, nf, nt, sky_size))
                    postf_dims = 2  ## time x npix

                    ## loop over pixels, interpolating the response as we go
                    ## only loop over pixels in the skymap with nonzero power
                    for i, response_sparse in zip(active_pixels_idx, responses_sparse):
                        if i in sm.mask_idx:
                            response = _interpolate(response_sparse)
                            integral[..., i] = response

                ## spherical harmonic search
                elif sm.basis == "sph":

                    alm_size = (sm.almax + 1) ** 2
                    ## angular coordinates of pixel indices
                    theta, phi = hp.pix2ang(params["nside"], active_pixels_idx)
                    Ylms = np.zeros((npix, alm_size), dtype="complex")
                    ## Get the spherical harmonics
                    for ii in range(alm_size):
                        lval, mval = sm.idxtoalm(sm.almax, ii)
                        Ylms[:, ii] = sph_harm_y(
                            mval, lval, theta, phi
                        )  ## theta, phi switched due to new scipy convention
                    ## check that the Ylms have the right number of pixels, sph terms
                    chex.assert_shape(Ylms, (npix, alm_size))

                    ## 3 x 3 x frequency x time x Ylms
                    integral = np.zeros((3, 3, nf, nt, alm_size), dtype="complex")
                    postf_dims = 2  ## time x Ylms

                    ## loop over pixels, interpolating the response as we go
                    for i, response_sparse in zip(active_pixels_idx, responses_sparse):
                        response = _interpolate(response_sparse)
                        integral += (
                            Ylms[None, None, None, None, i, :]
                            * response[..., None]
                            * dOmega
                        )
                else:
                    assert False

        elif sm.spatial_model_name == "isgwb":

            integral = np.zeros((3, 3, nf, nt))
            postf_dims = 1  ## just time

            ## loop over pixels, interpolating the response as we go
            for i, response_sparse in zip(active_pixels_idx, responses_sparse):
                response = _interpolate(response_sparse)
                integral += (1 / (4 * jnp.pi)) * response * dOmega

        else:
            assert False ## this should never happen

        ## TDI
        if params["tdi_lev"] == "xyz":
            mich_to_x1 = 4 * jnp.sin(freqs / FSTAR) ** 2
            integral = (
                integral
                * mich_to_x1[
                    np.newaxis, np.newaxis, :, *[np.newaxis for i in range(postf_dims)]
                ]
            )
        elif params["tdi_lev"] == "aet":
            # TODO
            raise NotImplementedError

        if plot_flag:
            sm.fdata_response_mat = integral
        else:
            sm.response_mat = integral
            sm.inj_response_mat = integral


def get_sparse_tf_grid(times, freqs):
    """
    Generate a sparse time-frequency grid suitable for interpolation.

    Parameters
    ----------
    times : array (ntimes,)
        dense time grid
    freqs : array (nfreqs,)
        dense frequency grid

    Returns
    -------
    array 1D
        sparse time grid
    array 1D
        sparse frequency grid
    """
    chex.assert_rank([times, freqs], 1)
    _ = times

    # approx 52 weeks, two per week, including endpoint. Used in periodic interpolation.
    # TODO optimize for times[-1] < YEAR
    times_sparse = jnp.linspace(0, YEAR, 105)

    # linear grid of step 1 mHz is largely enough (response variation insignificant)
    freqs_sparse = jnp.arange(freqs[0], freqs[-1], 1e-3)

    chex.assert_rank([times_sparse, freqs_sparse], 1)
    return times_sparse, freqs_sparse


def get_response_interpolator(times, freqs, times_sparse, freqs_sparse):
    """
    Generate interpolator function for response matrices.

    The interpolated response assumes 1-year periodicity for time.

    Parameters
    ----------
    times : array (nt,)
        dense (target) time grid
    freqs : array (nf,)
        dense (target) frequency grid
    times_sparse : array (nt_s,)
        sparse time grid
    freqs_sparse : array (nf_s,)
        sparse frequency grid

    Returns
    -------
    function
        an interpolator that receives a complex array of shape (3, 3, nf_s, nt_s) (the
        sparse response matrix) and returns a complex array (3, 3, nf, nt).

    Examples
    --------
    >>> t_sparse, f_sparse = get_sparse_tf_grid(times, freqs)
    >>> _interpolate = get_response_interpolator(times, freqs, t_sparse, f_sparse)
    >>> response = jax.jit(_interpolate)(response_sparse)
    """
    chex.assert_rank([times, freqs, times_sparse, freqs_sparse], 1)
    nt, nf, nt_s, nf_s = len(times), len(freqs), len(times_sparse), len(freqs_sparse)

    interp_vect = vmap(lambda r: jnp.interp(times, times_sparse, r, period=YEAR))
    interp_vecf = vmap(lambda r: jnp.interp(freqs, freqs_sparse, r))

    def interpolator(response_sparse):
        chex.assert_shape(response_sparse, (3, 3, nf_s, nt_s))

        rs_batched_t = response_sparse.reshape((-1, nt_s))
        response_sparse_f = interp_vect(rs_batched_t).reshape((3, 3, nf_s, nt))

        rsf_batched_f = response_sparse_f.transpose(0, 1, 3, 2).reshape((-1, nf_s))
        response = (
            interp_vecf(rsf_batched_f).reshape((3, 3, nt, nf)).transpose(0, 1, 3, 2)
        )

        chex.assert_shape(response, (3, 3, nf, nt))
        return response

    return interpolator


def mich_response_unconvolved(t, f, n, orbits):
    r"""
    Unconvolved Michelson (TDI gen 0) sky SGWB response.

    Parameters
    ----------
    t : float
        time
    f : float
        frequency
    n : array (3,)
        normalized vector in the direction of the GW source.
    orbits : tuple
        orbital information returned by compute_orbits().

    Returns
    -------
    complex array (3, 3)
        Unconvolved response matrix for the three data channels.

    Notes
    -----
    The quantity computed here is exactly

    .. math::

        \frac{1}{2} \sum_{A=+,\times}\left(F_I^A(f, \mathbf{n})^* F_J^A(f, \mathbf{n})
        \right)

    where :math:`I` and :math:`J` stand for TDI channels, and :math:`F_I^A` are antenna
    pattern functions.

    This is integrated against the sky map to produce the GW time-frequency correlation
    matrix.

    The convention here agrees with Criswell+25 eq. (6) (but for a complex conjugate),
    which is a corrected version of Banagiri+21 eq. (18).
    """
    chex.assert_shape([t, f], ())
    chex.assert_shape(n, (3,))

    res = jnp.zeros((3, 3), dtype=complex)

    # This loop intentionally uses python control flow so that it is
    # unrolled in tracing and the channels (c1, c2) are trace-time known.
    for c1 in range(3):
        for c2 in range(c1, 3):
            fp1 = mich_antenna_pattern(t, f, n, "plus", c1, orbits)
            fp2 = mich_antenna_pattern(t, f, n, "plus", c2, orbits)
            fc1 = mich_antenna_pattern(t, f, n, "cross", c1, orbits)
            fc2 = mich_antenna_pattern(t, f, n, "cross", c2, orbits)
            chex.assert_shape([fp1, fp2, fc1, fc2], ())
            res = res.at[c1, c2].set(0.5 * (fp1.conj() * fp2 + fc1.conj() * fc2))
            if c1 != c2:
                res = res.at[c2, c1].set(res[c1, c2].conj())

    chex.assert_shape(res, (3, 3))
    return res


def compute_orbits(times):
    """
    Compute orbit information at specified time array.

    Parameters
    ----------
    times : array 1D
        Times at which the positions of the spacecraft and link vectors should be
        computed.

    Returns
    -------
    array 1D
        the input times array
    array (3, ntimes, 3)
        Spacecraft positions in ecliptic cartesian coordinates as an array, where the
        first dimension specifies the spacecraft.
    array (6, ntimes, 3)
        Single link unit vectors in lisaorbits order.
    """
    chex.assert_rank(times, 1)

    # We are interfacing with non-jax-traceable numpy code:
    # therefore, compute all that we need from it upfront.
    orbits = lisaorbits.EqualArmlengthOrbits()
    sc_positions_icrs = jnp.array(
        orbits.compute_position(times, [1, 2, 3]).transpose(1, 0, 2)
    )
    link_vectors_icrs = jnp.array(orbits.compute_unit_vector(times).transpose(1, 0, 2))
    chex.assert_shape(sc_positions_icrs, (3, times.shape[0], 3))
    chex.assert_shape(link_vectors_icrs, (6, times.shape[0], 3))

    # v3 means ICRS equatorial coordinates. v2 => ecliptic.
    assert lisaorbits.__version__ >= "3"

    sc_positions = []
    link_vectors = []
    for i in range(3):
        sc_positions.append(_icrs_to_ecliptic(sc_positions_icrs[i]))
    for i in range(6):
        link_vectors.append(_icrs_to_ecliptic(link_vectors_icrs[i]))

    sc_positions = jnp.asarray(sc_positions)
    link_vectors = jnp.asarray(link_vectors)
    chex.assert_shape(sc_positions, (3, times.shape[0], 3))
    chex.assert_shape(link_vectors, (6, times.shape[0], 3))

    return (times, sc_positions, link_vectors)


def _icrs_to_ecliptic(positions_icrs):
    chex.assert_shape(positions_icrs, (None, 3))
    ntimes = positions_icrs.shape[0]

    x = positions_icrs[:, 0]
    y = positions_icrs[:, 1]
    z = positions_icrs[:, 2]
    coords = SkyCoord(
        x=x, y=y, z=z, unit="m", representation_type="cartesian", frame="icrs"
    )
    coords = coords.transform_to("barycentricmeanecliptic")
    coords.representation_type = "cartesian"
    x = jnp.array(coords.x / units.meter)  # pylint: disable=no-member
    y = jnp.array(coords.y / units.meter)  # pylint: disable=no-member
    z = jnp.array(coords.z / units.meter)  # pylint: disable=no-member

    positions_eclp = jnp.stack([x, y, z]).T
    chex.assert_shape(positions_eclp, (ntimes, 3))
    return positions_eclp


# The following traceable jax functions just look up values from the pre-computed arrays
# sc_positions and link_vectors.
def get_orbital_positions(t, orbits):
    """
    Look up S/C positions in orbits, in a way that is jax-traceable.

    This function does not perform interpolation. It will return the 'last known'
    position of the spacecraft.

    Parameters
    ----------
    t : float
        time.
    orbits : tuple
        orbital information returned by compute_orbits().

    Returns
    -------
    array (3, 3)
        The positions of the three spacecraft (first axis) in cartesian coordinates
        (second axis).
    """
    chex.assert_shape(t, ())
    times, sc_positions, _ = orbits
    chex.assert_rank(times, 1)
    chex.assert_shape(sc_positions, (3, times.shape[0], 3))

    idx = jnp.searchsorted(times, t)
    res = sc_positions[:, idx, :]
    chex.assert_shape(res, (3, 3))
    return res


def get_link_vectors(t, orbits):
    """
    Look up link unit vectors in orbits, in a way that is jax-traceable.

    This function does not perform interpolation. It will return the 'last known' unit
    vectors.

    Parameters
    ----------
    t : float
        time.
    orbits : tuple
        orbital information returned by compute_orbits().

    Returns
    -------
    array (6, 3)
        The unit vectors in 3D (second axis) for each of the single links (first axis)
        in lisaorbits order.
    """
    chex.assert_shape(t, ())
    times, _, link_vectors = orbits
    chex.assert_rank(times, 1)
    chex.assert_shape(link_vectors, (6, times.shape[0], 3))
    idx = jnp.searchsorted(times, t)
    res = link_vectors[:, idx, :]
    chex.assert_shape(res, (6, 3))
    return res


# Surprisingly, this does not exist in jax.scipy.special
def _sinc(x):
    # Inner select avoids NaN when differentiating at x=0
    _x = jnp.select([x != 0, True], [x, 1.0])
    return jnp.select([x != 0, True], [jnp.sin(_x) / _x, 1.0])


def timing_transfer_fn(f, costheta):
    """
    Timing transfer function for two-way photon propagation.

    Checked against Banagiri+21 eq (16) and Cornish & Rubbo 2003 eq (37). Also agrees
    with Romano & Cornish 2017 eq (5.27) up to a constant 2L/c. This seems due to the
    conversion between strain and timing measurements, eq (5.4) in the living review.

    Parameters
    ----------
    f : float
        frequency in Hz
    costheta : float
        cosine of angle between arm and sky direction

    Returns
    -------
    complex
        the transfer function.
    """
    chex.assert_shape([f, costheta], ())

    f0 = f / (2 * FSTAR)
    s1 = _sinc(f0 * (1 + costheta))
    s2 = _sinc(f0 * (1 - costheta))
    e1 = jnp.exp(-1j * f0 * (3 - costheta))
    e2 = jnp.exp(-1j * f0 * (1 - costheta))
    res = 0.5 * (s1 * e1 + s2 * e2)

    chex.assert_shape(res, ())
    return res


def mich_detector_tensor(f, u, v, n, r):
    """
    Michelson channel detector tensor.

    Checked against Banagiri+21 eq (15).

    Parameters
    ----------
    f : float
        frequency in Hz
    u : array (3,)
        normalized vector in the direction of the first arm
    v : array (3,)
        normalized vector in the direction of the second arm
    n : array (3,)
        normalized vector in the direction of the GW source
    r : array (3,)
        position of vertex S/C in barycentric ecliptic cartesian coordinates

    Returns
    -------
    complex array (3, 3)
        detector tensor
    """
    chex.assert_shape(f, ())
    chex.assert_shape([u, v, n, r], (3,))

    uu = jnp.outer(u, u)
    vv = jnp.outer(v, v)
    chex.assert_shape([uu, vv], (3, 3))
    un = jnp.dot(u, n)
    vn = jnp.dot(v, n)
    nr = jnp.dot(n, r)
    omega = 2 * jnp.pi * f
    chex.assert_shape([un, vn, nr, omega], ())

    tun = timing_transfer_fn(f, un)
    tvn = timing_transfer_fn(f, vn)
    chex.assert_shape([tun, tvn], ())

    # factor disagrees with Romano & Cornish (-1j -> +1j)
    factor = jnp.exp(-1j * omega * nr / CLIGHT)
    result = 0.5 * factor * (tun * uu - tvn * vv)

    chex.assert_shape(result, (3, 3))
    return result


def arm_orientations(t, sc, orbits):
    """
    Get unit vectors for left and right arm of a given spacecraft.

    The numbering convention is the same as in lisaorbits.

    Parameters
    ----------
    t : float
        time
    sc : int
        Spacecraft number (1, 2, 3). Must be known at JAX trace-time.
    orbits : tuple
        orbital information from compute_orbits().

    Returns
    -------
    array (3,)
        Unit vector pointing away from the given spacecraft (1 -> 3, 2 -> 1, 3 -> 2).
    array (3,)
        Unit vector pointing away from the given spacecraft (1 -> 2, 2 -> 3, 3 -> 1).
    """
    # sc=1,2,3 must be trace-time known
    chex.assert_shape([t, sc], ())
    assert 1 <= sc and sc <= 3

    # The lisaconstants indexing convention for links is receiver-sender.
    # In our case we want the directions of the links pointing away from
    # a given spacecraft.
    if sc == 1:
        idx_u, idx_v = 5, 2
        assert lisaorbits.LINKS[idx_u] == 21
        assert lisaorbits.LINKS[idx_v] == 31
    elif sc == 2:
        idx_u, idx_v = 4, 0
        assert lisaorbits.LINKS[idx_u] == 32
        assert lisaorbits.LINKS[idx_v] == 12
    else:
        idx_u, idx_v = 3, 1
        assert lisaorbits.LINKS[idx_u] == 13
        assert lisaorbits.LINKS[idx_v] == 23

    lv = get_link_vectors(t, orbits)
    chex.assert_shape(lv, (6, 3))
    u = lv[idx_u]
    v = lv[idx_v]

    chex.assert_shape([u, v], (3,))
    return u, v


def get_ortho_basis_ecliptic_3d(lam, beta):
    """
    Get right-handed orthonormal basis (n, l, m).

    This is the basis in Romano & Cornish 2017 eq (2.4).

    Parameters
    ----------
    lam : float
        ecliptic longitude
    beta : float
        ecliptic latitude

    Returns
    -------
    tuple (n, l, m)
        A tuple of arrays of shape (3,).
    """
    chex.assert_shape([lam, beta], ())

    theta, phi = jnp.pi / 2 - beta, lam

    ct, st = jnp.cos(theta), jnp.sin(theta)
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    enn = jnp.array([st * cp, st * sp, ct])
    ell = jnp.array([ct * cp, ct * sp, -st])
    emm = jnp.array([-sp, cp, 0])

    chex.assert_shape([enn, ell, emm], (3,))
    return enn, ell, emm


def mich_antenna_pattern(t, f, n, polarization: str, channel, orbits):
    """
    Compute Michelson (TDI gen 0) antenna pattern.

    Checked against Banagiri+21 and Romano & Cornish 2017.

    Parameters
    ----------
    t : float
        time
    f : float
        frequency
    n : array (3,)
        Unit vector in the direction of the GW source.
    polarization : str
        Should be "plus" or "cross". Must be known at JAX trace time.
    channel : int
        Channel index 0, 1, 2. Must be known at JAX trace time.
    orbits : tuple
        orbital information returned by compute_orbits().

    Returns
    -------
    complex
        The antenna pattern.
    """
    # polarization and channel must be trace-time known
    chex.assert_shape([t, f, channel], ())
    chex.assert_shape(n, (3,))
    assert polarization in ["plus", "cross"]
    assert 0 <= channel and channel < 3

    # lam, beta = ecliptic coordinates (lon, lat) of n
    n = n / jnp.linalg.norm(n)
    beta = jnp.arcsin(n[2])
    lam = jnp.atan2(n[1], n[0])
    lam = jnp.where(lam < 0, lam + 2 * jnp.pi, lam)

    # polarization tensor, Romano & Cornish 2017 eq 2.3
    _, ell, emm = get_ortho_basis_ecliptic_3d(lam, beta)
    if polarization == "plus":
        pol_tens = jnp.outer(ell, ell) - jnp.outer(emm, emm)
    else:
        pol_tens = jnp.outer(ell, emm) + jnp.outer(emm, ell)

    # detector tensor
    sc = channel + 1
    u, v = arm_orientations(t, sc, orbits)
    r = get_orbital_positions(t, orbits)[sc - 1]
    det_tens = mich_detector_tensor(f, u, v, n, r)

    chex.assert_shape([det_tens, pol_tens], (3, 3))
    res = jnp.tensordot(det_tens, pol_tens)
    chex.assert_shape(res, ())
    return res


def all_unit_vecs_healpix(nside):
    """
    Compute array of all unit vectors in the sky.

    Parameters
    ----------
    nside : int
        Healpix nside. Should be a power of 2.

    Returns
    -------
    array (npix, 3)
        Unit vectors for each healpix direction.
    """
    npix = hp.nside2npix(nside)
    ipix = jnp.arange(npix)
    x, y, z = hp.pix2vec(nside, ipix)
    res = jnp.asarray([x, y, z]).T
    chex.assert_shape(res, (npix, 3))
    return res


def _intarray_to_set(a: jax.Array) -> set:
    assert a.dtype == int
    return set([int(v) for v in a.ravel()])
