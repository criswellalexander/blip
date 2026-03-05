"""Sparse grids and interpolation."""

import chex
from jax import numpy as jnp, vmap

__all__ = ["get_sparse_tf_grid", "get_response_interpolator"]


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

    if len(times) == 1:
        times_sparse = times
    else:
        dt = times[1] - times[0]
        dt_s = 3600  # one hour
        if dt > dt_s:
            times_sparse = times
        else:
            times_sparse = jnp.arange(times[0], times[-1], dt_s)

    if len(freqs) == 1 or freqs[-1] > 10e-3:
        freqs_sparse = freqs
    else:
        df = freqs[1] = freqs[0]
        df_s = 1e-4  # 0.1 mHz
        if df > df_s:
            freqs_sparse = freqs
        else:
            freqs_sparse = jnp.arange(freqs[0], freqs[-1], df_s)

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

    interp_vect = vmap(lambda r: jnp.interp(times, times_sparse, r))
    interp_vecf = vmap(lambda r: jnp.interp(freqs, freqs_sparse, r))

    def interpolator(response_sparse):
        chex.assert_shape(response_sparse, (3, 3, nf_s, nt_s))

        # Global idea:
        #       interp time         interp freq
        # rs1 --------------> rs2 --------------> rs3.

        # interpolate magnitude and phase separately to avoid large magnitude errors.
        rs1 = response_sparse
        rs1_mag = jnp.abs(rs1)
        rs1_phs = jnp.angle(rs1)
        rs1_phs = jnp.unwrap(rs1_phs, axis=3)

        # rs1b = rs1, reshaped for interpolation along times
        rs1b_mag = rs1_mag.reshape((-1, nt_s))
        rs1b_phs = rs1_phs.reshape((-1, nt_s))

        rs2b_mag = interp_vect(rs1b_mag)
        rs2b_phs = interp_vect(rs1b_phs)

        rs2_mag = rs2b_mag.reshape((3, 3, nf_s, nt))
        rs2_phs = rs2b_phs.reshape((3, 3, nf_s, nt))
        rs2_phs = jnp.unwrap(rs2_phs, axis=2)

        # rs2c = rs2, reshaped for interpolation along frequencies
        rs2c_mag = rs2_mag.transpose(0, 1, 3, 2).reshape((-1, nf_s))
        rs2c_phs = rs2_phs.transpose(0, 1, 3, 2).reshape((-1, nf_s))

        rs3c_mag = interp_vecf(rs2c_mag)
        rs3c_phs = interp_vecf(rs2c_phs)

        rs3c = rs3c_mag * jnp.exp(1j * rs3c_phs)

        rs3 = rs3c.reshape((3, 3, nt, nf)).transpose(0, 1, 3, 2)

        chex.assert_shape(rs3, (3, 3, nf, nt))
        return rs3

    return interpolator
