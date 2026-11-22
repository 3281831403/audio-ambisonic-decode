"""Acceptance tests for ``ambi.decode`` -- 15 tests, all initially red.

Conventions
-----------
* ACN channel order: ``[W, Y, Z, X]``.
* Right-handed listener frame: ``+X`` forward, ``+Y`` left, ``+Z`` up.
* Azimuth and elevation are in radians.
* N3D order-1 basis: ``W = 1``, ``[Y, Z, X] = sqrt(3) * (y, z, x)`` for a
  unit direction ``(x, y, z)``.
* SN3D divides order 1 by ``sqrt(3)``.
* Absolute tolerance is ``1e-9`` everywhere.

The implementation in ``ambi/decode.py`` consists of eight
``NotImplementedError`` stubs, so the initial run is exactly ``15 failed``
with no collection, import or syntax errors.  This file is the acceptance
specification and must not be modified while fixing the decoder.
"""

from __future__ import annotations

import math

import pytest

from ambi.decode import (
    apply_matrix,
    decode_batch,
    decode_matrix,
    energy_ratio,
    rotate,
    sh_basis,
    to_sn3d,
    validate_basis,
)

SQRT3 = math.sqrt(3.0)
TETRA_EL = math.atan(1.0 / math.sqrt(2.0))
CUBE_EL = math.atan(1.0 / math.sqrt(2.0))

# Four vertices of a regular tetrahedron: a proper 3-D, order-1 layout.
TETRA = [
    (math.pi / 4.0, TETRA_EL),
    (-math.pi / 4.0, -TETRA_EL),
    (3.0 * math.pi / 4.0, -TETRA_EL),
    (-3.0 * math.pi / 4.0, TETRA_EL),
]

# Eight vertices of a cube: a uniform 3-D layout for the energy tests.
CUBE = [
    (azimuth, elevation)
    for elevation in (CUBE_EL, -CUBE_EL)
    for azimuth in (
        math.pi / 4.0,
        3.0 * math.pi / 4.0,
        5.0 * math.pi / 4.0,
        7.0 * math.pi / 4.0,
    )
]

# Six axis directions: a tiny exact spherical design for Gram checks.
AXES = [
    (0.0, 0.0),
    (math.pi, 0.0),
    (math.pi / 2.0, 0.0),
    (-math.pi / 2.0, 0.0),
    (0.0, math.pi / 2.0),
    (0.0, -math.pi / 2.0),
]


def _assert_vector_close(actual, expected, tol=1e-9):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got == pytest.approx(want, abs=tol)


# ---------------------------------------------------------------------------
# One test per documented defect (10)
# ---------------------------------------------------------------------------


def test_sh_basis_separates_y_and_z():
    """Defect 1: Y must use sin(azimuth), Z must use sin(elevation)."""
    left = sh_basis(1, math.pi / 2.0, 0.0)
    up = sh_basis(1, 0.0, math.pi / 2.0)

    assert left[1] == pytest.approx(SQRT3, abs=1e-9)
    assert left[2] == pytest.approx(0.0, abs=1e-9)
    assert left[3] == pytest.approx(0.0, abs=1e-9)

    assert up[1] == pytest.approx(0.0, abs=1e-9)
    assert up[2] == pytest.approx(SQRT3, abs=1e-9)
    assert up[3] == pytest.approx(0.0, abs=1e-9)


def test_sh_basis_uses_n3d_normalization():
    """Defect 2: order 1 must carry the N3D sqrt(3) factor."""
    azimuth = 0.37
    elevation = 0.21
    basis = sh_basis(1, azimuth, elevation)
    horizontal = math.cos(elevation)

    assert basis == pytest.approx(
        [
            1.0,
            SQRT3 * horizontal * math.sin(azimuth),
            SQRT3 * math.sin(elevation),
            SQRT3 * horizontal * math.cos(azimuth),
        ],
        abs=1e-9,
    )
    assert sum(value * value for value in basis[1:]) == pytest.approx(
        3.0, abs=1e-9
    )


def test_to_sn3d_preserves_w():
    """Defect 3: W is degree 0 and must not be divided by sqrt(3)."""
    n3d = [4.0, 2.0, -1.0, 3.0]
    sn3d = to_sn3d(n3d)

    assert len(sn3d) == 4
    assert sn3d[0] == pytest.approx(4.0, abs=1e-9)
    assert sn3d[1] == pytest.approx(2.0 / SQRT3, abs=1e-9)
    assert sn3d[2] == pytest.approx(-1.0 / SQRT3, abs=1e-9)
    assert sn3d[3] == pytest.approx(3.0 / SQRT3, abs=1e-9)

    assert to_sn3d([2.5]) == pytest.approx([2.5], abs=1e-9)


def test_decode_matrix_is_soundfield_to_speakers():
    """Defect 4: a transposed decoder mirrors every speaker direction."""
    matrix = decode_matrix(len(TETRA), TETRA, 1)

    for index, (azimuth, elevation) in enumerate(TETRA):
        speakers = apply_matrix(sh_basis(1, azimuth, elevation), matrix)
        loudest = max(
            range(len(speakers)), key=lambda position: speakers[position]
        )

        assert loudest == index, (
            f"a source at tetrahedron vertex {index} was decoded loudest at "
            f"speaker {loudest}"
        )
        assert speakers[index] > 0.5


def test_decode_matrix_applies_max_re_weighting():
    """Defect 5: without max-rE the first-order part overwhelms W."""
    matrix = decode_matrix(len(CUBE), CUBE, 1)

    for row in matrix:
        w_gain = abs(row[0])
        first_order_gain = math.sqrt(sum(value * value for value in row[1:]))

        assert first_order_gain > 0.0
        assert w_gain >= 0.9 * first_order_gain, (
            "max-rE must attenuate order 1 relative to W; got "
            f"W={w_gain!r}, first-order={first_order_gain!r}"
        )


def test_apply_matrix_conserves_energy_across_speaker_counts():
    """Defect 6: total energy must not change with the number of speakers."""
    field = [1.0, 0.5, -0.25, 0.75]
    expected_energy = sum(value * value for value in field)

    for layout in (TETRA, CUBE):
        matrix = decode_matrix(len(layout), layout, 1)
        speakers = apply_matrix(field, matrix)
        energy = sum(value * value for value in speakers)

        assert len(speakers) == len(layout)
        assert energy == pytest.approx(expected_energy, abs=1e-9)


def test_rotate_keeps_w_invariant():
    """Defect 7: rotating the field must not modulate the omnidirectional W."""
    field = [1.0, 0.2, -0.3, 0.4]
    rotated = rotate(field, 0.8, 0.3)

    assert rotated[0] == pytest.approx(1.0, abs=1e-9)
    assert math.sqrt(sum(value * value for value in rotated[1:])) == pytest.approx(
        math.sqrt(sum(value * value for value in field[1:])), abs=1e-9
    )

    front_to_left = rotate(sh_basis(1, 0.0, 0.0), math.pi / 2.0, 0.0)
    assert front_to_left[0] == pytest.approx(1.0, abs=1e-9)
    assert front_to_left[1] == pytest.approx(SQRT3, abs=1e-9)
    assert front_to_left[2] == pytest.approx(0.0, abs=1e-9)
    assert front_to_left[3] == pytest.approx(0.0, abs=1e-9)


def test_energy_ratio_returns_plane_wave_half():
    """Defect 8: W**2 / raw total is only 0.25, always below one third."""
    plane_wave = sh_basis(1, 0.3, -0.2)

    assert energy_ratio(plane_wave) == pytest.approx(0.5, abs=1e-9)
    assert energy_ratio([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-9)
    assert energy_ratio([0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_validate_basis_detects_non_orthogonality():
    """Defect 9: a matching channel count is not proof of an orthogonal basis."""
    valid = [sh_basis(1, azimuth, elevation) for azimuth, elevation in AXES]
    assert validate_basis(valid, 1) == []

    # Keep the channel count and every channel's mean square, but make Y a
    # copy of X.  The sampled X/Y inner product is now 1 instead of 0.
    non_orthogonal = [
        [sample[0], sample[3], sample[2], sample[3]] for sample in valid
    ]
    errors = validate_basis(non_orthogonal, 1)

    assert errors
    assert any("orth" in error.lower() for error in errors)


def test_decode_batch_does_not_reuse_rotated_state():
    """Defect 10: frame 2 must not start from frame 1's rotated state."""
    config = {"layout": TETRA, "order": 1, "rotation": (math.pi / 2.0, 0.0)}
    front = sh_basis(1, 0.0, 0.0)
    left = sh_basis(1, math.pi / 2.0, 0.0)

    batch = decode_batch([front, left], config)
    matrix = decode_matrix(len(TETRA), TETRA, 1)

    expected_front = apply_matrix(rotate(front, math.pi / 2.0, 0.0), matrix)
    expected_left_once = apply_matrix(rotate(left, math.pi / 2.0, 0.0), matrix)

    _assert_vector_close(batch["signals"][0], expected_front)
    _assert_vector_close(batch["signals"][1], expected_left_once)

    single_left = decode_batch([left], config)["signals"][0]
    _assert_vector_close(batch["signals"][1], single_left)


# ---------------------------------------------------------------------------
# Additional invariants (5)
# ---------------------------------------------------------------------------


def test_front_direction_is_w1_x1_and_yz_zero():
    """At the front, W=1 and the normalised direction cosine on X is 1."""
    basis = sh_basis(1, 0.0, 0.0)

    assert basis[0] == pytest.approx(1.0, abs=1e-9)
    assert basis[1] == pytest.approx(0.0, abs=1e-9)
    assert basis[2] == pytest.approx(0.0, abs=1e-9)
    # N3D stores sqrt(3) * direction cosine, so X / sqrt(3) is the unit X.
    assert basis[3] / SQRT3 == pytest.approx(1.0, abs=1e-9)


def test_sh_basis_is_orthogonal_on_uniform_sphere():
    """Uniformly sampled N3D harmonics have an identity Gram matrix."""
    samples = [sh_basis(1, azimuth, elevation) for azimuth, elevation in AXES]
    sample_count = len(samples)

    for first in range(4):
        for second in range(4):
            product = sum(
                sample[first] * sample[second] for sample in samples
            )
            expected = float(sample_count) if first == second else 0.0
            assert product == pytest.approx(expected, abs=1e-9), (
                f"Gram entry ({first}, {second}) was {product!r}, "
                f"expected {expected!r}"
            )


def test_decode_matrix_rows_have_unit_energy():
    """Every decoder row must be a unit-energy speaker feed."""
    for layout in (TETRA, CUBE):
        matrix = decode_matrix(len(layout), layout, 1)
        assert len(matrix) == len(layout)

        for row in matrix:
            assert len(row) == 4
            assert sum(value * value for value in row) == pytest.approx(
                1.0, abs=1e-9
            )


def test_rotate_360_degrees_returns_input():
    """A full turn is the identity for all four channels."""
    field = [1.0, -0.4, 0.2, 0.7]
    rotated = rotate(field, 2.0 * math.pi, 0.0)

    _assert_vector_close(rotated, field)


def test_decode_batch_matches_individual_decodes():
    """Two frames decoded together must equal two independent decodes."""
    config = {"layout": TETRA, "order": 1}
    frames = [
        sh_basis(1, 0.0, 0.0),
        sh_basis(1, math.pi / 2.0, 0.0),
    ]

    batch = decode_batch(frames, config)
    matrix = decode_matrix(len(TETRA), TETRA, 1)

    assert batch["order"] == 1
    assert len(batch["matrix"]) == len(matrix)
    for actual_row, expected_row in zip(batch["matrix"], matrix):
        _assert_vector_close(actual_row, expected_row)
    assert len(batch["signals"]) == len(frames)
    assert len(batch["energies"]) == len(frames)

    for index, frame in enumerate(frames):
        expected = apply_matrix(frame, matrix)
        _assert_vector_close(batch["signals"][index], expected)
        _assert_vector_close(
            batch["signals"][index],
            decode_batch([frame], config)["signals"][0],
        )
        assert batch["energies"][index] == pytest.approx(
            sum(value * value for value in expected), abs=1e-9
        )
