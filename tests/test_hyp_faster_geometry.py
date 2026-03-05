import pytest
from hypothesis import given, example, note, strategies as st

import jax
import chex
from jax import numpy as jnp, vmap, jit

from lisaconstants import ASTRONOMICAL_YEAR as YEAR

from blip.src.faster_geometry import (
    get_response_interpolator,
    mich_response_unconvolved,
    compute_orbits,
    get_ortho_basis_ecliptic_3d,
    all_unit_vecs_healpix,
    arm_orientations,
    ARMLENGTH,
)

jax.config.update("jax_enable_x64", True)

# vectorize response on (t, f)
_mru_vect = vmap(mich_response_unconvolved, (0, None, None, None), out_axes=2)
_mru_vec2 = vmap(_mru_vect, (None, 0, None, None), out_axes=2)
mru_vec2 = jit(_mru_vec2)
mru = jit(mich_response_unconvolved)
mru_vecn = jit(vmap(mich_response_unconvolved, (None, None, 0, None)))


@st.composite
def interp_patch(draw):
    # We are going to guarantee a certain precision only
    # up to fmax
    fmax = 10e-3

    # size of sparse grid (tune this)
    dt_s = 12 * 3600
    df_s = 1e-4

    # rectangle of sparse grid points
    t0 = draw(st.floats(0, 1.5 * YEAR))
    t1 = t0 + dt_s
    f0 = draw(st.floats(df_s, fmax - df_s))
    f1 = f0 + df_s

    # point to be interpolated
    t = draw(st.floats(t0, t1))
    f = draw(st.floats(f0, f1))

    return (t0, t, t1), (f0, f, f1)


@st.composite
def direction(draw):
    theta = draw(st.floats(0, jnp.pi))
    phi = draw(st.floats(0, 2 * jnp.pi))
    cost, sint = jnp.cos(theta), jnp.sin(theta)
    cosp, sinp = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([sint * cosp, sint * sinp, cost])


@example(
    ((0.0, 1.0, 3600.0), (0.0078125, 0.0078125, 0.0079125)),
    jnp.array([0.47942554, 0.0, 0.87758256]),
)
@example(
    (
        (0.0, 0.0, 3600.0),
        (0.0005835943283123456, 0.0006474497657475774, 0.0006835943283123457),
    ),
    jnp.array([0.90352311, 0.0, -0.42853936]),
)
@example(
    ((42601709.0, 42605200.0, 42605309.0), (0.001953125, 0.001953125, 0.002053125)),
    jnp.array([0.9026844, 0.0, -0.43030323]),
)
@example(
    ((21355701.0, 21394281.0, 21398901.0), (0.0078125, 0.0078125, 0.0079125)),
    jnp.array([0.0, 0.0, 1.0]),
)
@pytest.mark.skip
@given(interp_patch(), direction())
def test_interpolation(patch, n):
    (t0, t, t1), (f0, f, f1) = patch

    note(f"t - t0 = {(t - t0)/3600:.1f} hours")
    note(f"{(t-t0)/(t1-t0) = }")
    note(f"f - f0 = {(f - f0)/1e-3:.2f} mHz")
    note(f"{(f-f0)/(f1-f0) = }")

    times = jnp.array([t0, t, t1])
    freqs = jnp.array([f0, f, f1])
    times_s = jnp.array([t0, t1])
    freqs_s = jnp.array([f0, f1])

    orbits = compute_orbits(times)
    orbits_s = compute_orbits(times_s)

    # get reference "correct" response
    resp = mru_vec2(times, freqs, n, orbits)
    chex.assert_shape(resp, (3, 3, 3, 3))
    assert resp.dtype == jnp.complex128

    # get response from interpolation
    resp_s = mru_vec2(times_s, freqs_s, n, orbits_s)
    _interpolate = get_response_interpolator(times, freqs, times_s, freqs_s)
    resp_interp = _interpolate(resp_s)
    chex.assert_shape(resp_interp, (3, 3, 3, 3))
    assert resp_interp.dtype == jnp.complex128

    #### compare ####

    # no interpolation at all for borders
    assert jnp.allclose(resp[:, :, 0, 0], resp_interp[:, :, 0, 0])
    assert jnp.allclose(resp[:, :, 0, 2], resp_interp[:, :, 0, 2])
    assert jnp.allclose(resp[:, :, 2, 0], resp_interp[:, :, 2, 0])
    assert jnp.allclose(resp[:, :, 2, 2], resp_interp[:, :, 2, 2])

    # in the middle, interpolation should give reasonable results
    maxabs = jnp.max(jnp.abs(resp[:, :, 1, 1]))
    absdiff = jnp.max(jnp.abs(resp - resp_interp)[:, :, 1, 1])
    reldiff = jnp.max(jnp.abs((resp - resp_interp) / resp)[:, :, 1, 1])
    note(f"max rel diff = {reldiff:.2e}")
    note(f"max abs diff / max abs resp = {absdiff/maxabs:.2e}")
    note(f"{resp[:,:,1,1] = }")
    note(f"{resp_interp[:,:,1,1] = }")
    assert absdiff < 1e-4 * maxabs

    # NOTE this test shows that accurate interpolation needs a sparse grid with time and
    # frequency steps that are impractically small, which is why interpolation is now
    # disabled by default.


@given(st.floats(0, YEAR), st.floats(1e-6, 0.1), direction(), st.integers(1, 2))
def test_response_symmetries(t, f, n, roll):
    orbits = compute_orbits(jnp.array([t]))
    response = mru(t, f, n, orbits)

    # result is real and positive
    for c in range(3):
        assert jnp.allclose(response[c, c].imag, 0.0)
        assert jnp.all(response[c, c].real >= 0)

    # hermitean
    for c1 in range(3):
        for c2 in range(3):
            assert jnp.allclose(response[c1, c2], response[c2, c1].conj())

    # swap spacecraft => swap xyz channels
    orbits_swap = compute_orbits(jnp.array([t]), betaphase=-2 * jnp.pi / 3 * roll)
    response_swap = mru(t, f, n, orbits_swap)
    assert jnp.allclose(response_swap, jnp.roll(response, roll, (0, 1)))


@given(st.floats(-jnp.pi / 2, jnp.pi / 2), st.floats(0, 2 * jnp.pi))
def test_ortho_basis(beta, lam):
    enn, ell, emm = get_ortho_basis_ecliptic_3d(lam, beta)
    assert jnp.isclose(jnp.linalg.norm(enn), 1.0)
    assert jnp.isclose(jnp.linalg.norm(ell), 1.0)
    assert jnp.isclose(jnp.linalg.norm(emm), 1.0)
    assert jnp.isclose(jnp.dot(enn, ell), 0.0)
    assert jnp.isclose(jnp.dot(ell, emm), 0.0)
    assert jnp.isclose(jnp.dot(emm, enn), 0.0)
    assert jnp.allclose(jnp.cross(enn, ell), emm)


@given(st.floats(0, YEAR))
def test_arm_orientations(t):
    times = jnp.array([t])
    orbits = compute_orbits(times)
    for sc in range(1, 4):
        u, v = arm_orientations(t, sc, orbits)
        chex.assert_shape([u, v], (3,))
        uv = jnp.dot(u, v)
        # cos(60 deg) = 1/2
        assert jnp.allclose(uv, 0.5)

    u1, v1 = arm_orientations(t, 1, orbits)
    u2, v2 = arm_orientations(t, 2, orbits)
    u3, v3 = arm_orientations(t, 3, orbits)
    assert jnp.allclose(v1, -u3)
    assert jnp.allclose(u1, -v2)
    assert jnp.allclose(u2, -v3)


@given(st.floats(0, YEAR))
def test_armlength(t):
    times = jnp.array([t])
    orbits = compute_orbits(times)

    _, sc_pos, _ = orbits
    assert jnp.isclose(jnp.linalg.norm(sc_pos[0, 0] - sc_pos[1, 0]), ARMLENGTH)
    assert jnp.isclose(jnp.linalg.norm(sc_pos[1, 0] - sc_pos[2, 0]), ARMLENGTH)
    assert jnp.isclose(jnp.linalg.norm(sc_pos[2, 0] - sc_pos[0, 0]), ARMLENGTH)


@given(st.floats(0, YEAR))
def test_low_freq_limit(t):
    orbits = compute_orbits(jnp.array([t]))
    f = 1e-6
    allsky = all_unit_vecs_healpix(nside=8)
    response = mru_vecn(t, f, allsky, orbits)
    chex.assert_shape(response, (allsky.shape[0], 3, 3))

    # normalization: sky- and polarization-average should be 3/20 in low-freq limit
    for c in range(3):
        assert jnp.allclose(response[:, c, c].mean(), 3 / 20, atol=1e-4)

    # for off-diagonal elements, requirement that T response is zero => (xy) = -(xx)/2
    for c1 in range(3):
        for c2 in range(c1 + 1, 3):
            note(f"{c1=}, {c2=}")
            assert jnp.allclose(response[:, c1, c2].mean(), -3 / 40, atol=1e-4)
