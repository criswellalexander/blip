import numpy as np
import healpy as hp
import pytest
from jax import numpy as jnp, vmap, jit
import chex

from blip.src.makeLISAdata import sqrt_psd
from blip.src.faster_geometry.core import mich_response_unconvolved
from blip.src.faster_geometry.orbit import compute_orbits
from blip.src.faster_geometry.util import get_vecs_all_sky
from blip.src.faster_geometry.const import FSTAR
from blip.src import astro


def _legacy_cholesky_symm(m):
    '''

    Reference copy of the previous data-generation decomposition (removed from makeLISAdata).

    Only valid for response matrices of the channel-symmetric form [[A,B,B],[B*,A,B],[B*,B*,A]],
    which holds for isotropic skies. Kept here to check that isotropic injections are unchanged.

    Arguments
    ------------------------------
    m : complex array (3, 3, ...)
    	Response matrix, with the two TDI channel axes first.

    Returns
    ------------------------------
    chol : complex array (3, 3, ...)
    	Lower-triangular matrix L such that L L^H is the channel-symmetrized m.

    '''
    a = jnp.abs(jnp.mean(jnp.array([m[0, 0], m[1, 1], m[2, 2]]), axis=0))
    b = jnp.mean(jnp.array([m[0, 1], m[0, 2], m[1, 2]]), axis=0)
    mu = jnp.sqrt(a)
    lam = b.conj() / mu
    rho = lam
    sigma = jnp.sqrt(jnp.maximum(0.0, a - jnp.abs(lam) ** 2))
    kappa = (b.conj() - jnp.abs(lam) ** 2) / sigma
    beta = jnp.sqrt(jnp.maximum(0.0, a - jnp.abs(kappa) ** 2 - jnp.abs(lam) ** 2))
    _z = 0.0 * mu
    return jnp.array([[mu, _z, _z], [lam, sigma, _z], [rho, kappa, beta]])


def _llh(L):
    '''

    Compute L L^H for a stack of matrices with the channel axes first.

    Arguments
    ------------------------------
    L : complex array (3, 3, ...)
    	Matrix factor, with the two TDI channel axes first.

    Returns
    ------------------------------
    llh : complex array (3, 3, ...)
    	The product L L^H, same shape as L.

    '''
    return jnp.einsum('ik...,jk...->ij...', L, L.conj())


def _xyz_response(skymap, nside=4):
    '''

    Sky-integrated XYZ response for a pixel skymap at a few frequencies and times.

    Mirrors faster_geometry's _do_fixed_submodel (without interpolation) for a small grid.

    Arguments
    ------------------------------
    skymap : numpy.ndarray
    	Healpix map of length npix; normalized here so that its sky integral is 1.
    nside : int
    	Healpix nside of the skymap. (Default: 4)

    Returns
    ------------------------------
    response : complex array (3, 3, nf, nt)
    	Skymap-convolved XYZ response matrix.

    '''
    dOmega = hp.nside2pixarea(nside)
    skymap = skymap / (np.sum(skymap) * dOmega)
    idx = np.flatnonzero(skymap)
    vecs = jnp.array(get_vecs_all_sky(nside)[idx])
    ## a few orbital phases over a year and frequencies spanning the foreground band
    times = jnp.linspace(0, 3.15e7, 3, endpoint=False)
    freqs = jnp.array([2e-4, 3e-3])
    orbits = compute_orbits(times)
    ## vectorize over times, frequencies, then pixels -> (npix, nf, nt, 3, 3)
    mru = jit(vmap(vmap(vmap(mich_response_unconvolved, (0, None, None, None)),
                        (None, 0, None, None)), (None, None, 0, None)))
    R = np.array(mru(times, freqs, vecs, orbits))
    R = np.einsum('pftij,p->ijft', R, skymap[idx] * dOmega)
    ## Michelson -> XYZ, as in faster_geometry._mich_to_tdi
    return R * np.array(4 * jnp.sin(freqs / FSTAR) ** 2)[None, None, :, None]


@pytest.fixture(scope="module")
def iso_response():
    return _xyz_response(np.ones(hp.nside2npix(4)))


@pytest.fixture(scope="module")
def galaxy_response():
    nside = 4
    skymap = astro.generate_galactic_foreground(2.9, 0.3, nside)
    skymap = skymap * (skymap > np.e**-4 * np.max(skymap))
    return _xyz_response(skymap, nside=nside)


def test_sqrt_psd_symmetric():
    a = jnp.array([2.0, 1.0])
    b = jnp.array([0.1 + 0.46j, -0.5])
    bbar = b.conj()

    m = jnp.array(
        [
            [a, b, b],
            [bbar, a, b],
            [bbar, bbar, a],
        ]
    )

    L = sqrt_psd(m)
    chex.assert_shape(L, m.shape)
    assert jnp.all(jnp.isfinite(L))
    assert jnp.allclose(_llh(L), m)


def test_sqrt_psd_general_hermitian():
    ## random full-rank Hermitian PSD matrices with no channel symmetry
    rng = np.random.default_rng(0)
    g = rng.normal(size=(4, 3, 3)) + 1j * rng.normal(size=(4, 3, 3))
    m = np.moveaxis(g @ g.conj().transpose(0, 2, 1), 0, -1)

    L = sqrt_psd(m)
    assert jnp.all(jnp.isfinite(L))
    assert jnp.allclose(_llh(L), m)


def test_sqrt_psd_rank2():
    ## rank-2 matrix with (1,1,1) null vector, like the low-frequency XYZ GW response
    rng = np.random.default_rng(1)
    g = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    g = g - g.mean(axis=0, keepdims=True)
    m = (g @ g.conj().T)[:, :, None]

    L = sqrt_psd(m)
    assert jnp.all(jnp.isfinite(L))
    assert jnp.allclose(_llh(L), m)


def test_sqrt_psd_galaxy(galaxy_response):
    ## anisotropic response breaks channel symmetry; the factor must still reproduce it exactly
    L = sqrt_psd(galaxy_response)
    assert jnp.all(jnp.isfinite(L))
    np.testing.assert_allclose(_llh(L), galaxy_response, atol=1e-10 * np.abs(galaxy_response).max())

    ## sanity check that this case actually discriminates from the legacy symmetrized form
    legacy = _llh(_legacy_cholesky_symm(galaxy_response))
    assert not np.allclose(legacy, galaxy_response, rtol=1e-2)


def test_sqrt_psd_isotropic_unchanged(iso_response):
    ## a truly isotropic response has the channel-symmetric form [[A,B,B],[B*,A,B],[B*,B*,A]];
    ## build it from the computed response, as the legacy method assumed
    a = np.mean([iso_response[0, 0], iso_response[1, 1], iso_response[2, 2]], axis=0)
    b = np.mean([iso_response[0, 1], iso_response[0, 2], iso_response[1, 2]], axis=0)
    m_sym = np.array([[a, b, b], [b.conj(), a, b], [b.conj(), b.conj(), a]])

    ## on the symmetric form, new and legacy factors give identical covariances
    L_new = sqrt_psd(m_sym)
    L_old = _legacy_cholesky_symm(m_sym)
    atol = 1e-10 * np.abs(m_sym).max()
    np.testing.assert_allclose(_llh(L_new), m_sym, atol=atol)
    np.testing.assert_allclose(_llh(L_old), m_sym, atol=atol)
    np.testing.assert_allclose(_llh(L_new), _llh(L_old), atol=atol)

    ## the pixelized isotropic response deviates from exact channel symmetry only at the
    ## discretization level (~0.2% at nside=4); the new factor reproduces it exactly
    np.testing.assert_allclose(_llh(sqrt_psd(iso_response)), iso_response, atol=atol)
    np.testing.assert_allclose(_llh(_legacy_cholesky_symm(iso_response)), iso_response,
                               atol=5e-3 * np.abs(iso_response).max())


def test_sqrt_psd_isotropic_sample_covariance(iso_response):
    ## distributional check: coloured draws from both factors converge to the same covariance
    m = np.asarray(iso_response[:, :, 0, 0])
    ndraw = 100000
    rng = np.random.default_rng(42)
    z = rng.normal(size=(3, ndraw)) + 1j * rng.normal(size=(3, ndraw))
    for L in [sqrt_psd(m[:, :, None])[:, :, 0], _legacy_cholesky_symm(m[:, :, None])[:, :, 0]]:
        d = np.asarray(L) @ z
        ## standard complex normal has <z z^*> = 2
        cov = d @ d.conj().T / (2 * ndraw)
        np.testing.assert_allclose(cov, m, atol=0.02 * np.abs(m).max())
