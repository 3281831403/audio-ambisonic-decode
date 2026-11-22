"""First-order Ambisonic decoding primitives.

Conventions used by every function in this module:

* ACN channel order for orders 0 and 1 is ``[W, Y, Z, X]``.
* The listener frame is right-handed: ``+X`` forward, ``+Y`` left, ``+Z`` up.
* ``azimuth`` is in radians, counter-clockwise from ``+X`` toward ``+Y``.
* ``elevation`` is in radians above the horizontal plane.
* Order-1 coefficients use the N3D normalisation
  ``W = 1``, ``Y = sqrt(3) * y``, ``Z = sqrt(3) * z``, ``X = sqrt(3) * x``
  for a unit direction ``(x, y, z)``.
* Numerical assertions in the acceptance suite use an absolute tolerance of
  ``1e-9``.
"""

from __future__ import annotations

import math

__all__ = [
    "sh_basis",
    "to_sn3d",
    "decode_matrix",
    "apply_matrix",
    "rotate",
    "energy_ratio",
    "validate_basis",
    "decode_batch",
]

_TOL = 1e-9


def sh_basis(order: int, azimuth: float, elevation: float) -> list[float]:
    """Return real N3D spherical harmonics through first order.

    Channel order is ACN, ``[W, Y, Z, X]``.  For a unit direction

    ``x = cos(elevation) * cos(azimuth)``,
    ``y = cos(elevation) * sin(azimuth)``,
    ``z = sin(elevation)``

    the coefficients are::

        W = 1
        Y = sqrt(3) * y
        Z = sqrt(3) * z
        X = sqrt(3) * x

    The ``sqrt(3)`` factors are the N3D per-order normalisation.  Consequently
    a single direction has ``W**2 + Y**2 + Z**2 + X**2 == 4``; at the front
    (``azimuth == 0``, ``elevation == 0``) the N3D coefficients are
    ``[1, 0, 0, sqrt(3)]``, i.e. the normalised direction cosine on X is one.

    ``order`` must be 0 (return ``[W]``) or 1 (return ``[W, Y, Z, X]``).
    Components must match the formulas above to within ``1e-9``.
    """
    if order == 0:
        return [1.0]
    if order != 1:
        raise ValueError("order must be 0 or 1")

    horizontal = math.cos(elevation)
    x = horizontal * math.cos(azimuth)
    # Y tracks the azimuth (left/right); Z tracks the elevation (up/down).
    y = horizontal * math.sin(azimuth)
    z = math.sin(elevation)

    return [1.0, math.sqrt(3.0) * y, math.sqrt(3.0) * z, math.sqrt(3.0) * x]


def to_sn3d(ambi: list[float]) -> list[float]:
    """Convert first-order N3D coefficients to SN3D.

    SN3D divides each order by ``sqrt(2 * degree + 1)``.  Thus W (degree 0) is
    unchanged, while Y, Z and X (degree 1) are each divided by ``sqrt(3)``.
    W must not be attenuated: it is the omnidirectional reference channel.

    ``ambi`` may contain one channel (W only) or four channels in ACN order.
    Return a new list; do not mutate the input.  Results must agree to within
    ``1e-9``.
    """
    result = list(ambi)
    if len(result) == 1:
        # Degree 0 only: W is the omnidirectional reference and stays as is.
        return result
    # Degree 0 is untouched; every degree-1 channel is divided by sqrt(3).
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    for channel in range(1, len(result)):
        result[channel] *= inv_sqrt3
    return result


def decode_matrix(
    n_speakers: int,
    layout: list[tuple[float, float]],
    order: int,
) -> list[list[float]]:
    """Build a sound-field-to-speaker decoder for a 3-D speaker layout.

    ``layout`` is a sequence of ``(azimuth, elevation)`` pairs in radians, one
    per speaker; ``n_speakers`` must equal ``len(layout)``.  The returned matrix
    has one row per speaker and one column per ACN channel, so it maps an
    ambisonic coefficient vector to speaker signals.  A transposed matrix is an
    encoder (speakers to sound field), not a decoder, and mirrors directions.

    The projection decoder is built from the sampled basis ``Y``::

        D0[r][c] = Y[r][c] / N
        D[r][c]  = D0[r][c] * g[c] / ||D0[r] * g||

    where the max-rE weights are ``g = [1, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]``
    for order 1 and ``g = [1]`` for order 0.  The per-row normalisation makes
    every row's sum of squares exactly one (unit row energy), while the
    first-order weight keeps the decoder from becoming too directional.  The
    layout must provide at least ``(order + 1) ** 2`` speakers.

    ``order`` must be 0 or 1.  Matrix entries must be accurate to ``1e-9``.
    """
    if n_speakers != len(layout):
        raise ValueError("n_speakers must equal len(layout)")
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")
    n_channels = (order + 1) ** 2
    if n_speakers < n_channels:
        raise ValueError(
            "layout must provide at least (order + 1) ** 2 speakers"
        )

    if order == 0:
        weights = [1.0]
    else:
        first_order_weight = 1.0 / math.sqrt(3.0)
        weights = [1.0, first_order_weight, first_order_weight, first_order_weight]

    # Rows are speakers, columns are sound-field channels: this orientation is
    # sound-field -> speakers.  Transposing would encode instead of decoding.
    matrix: list[list[float]] = []
    for azimuth, elevation in layout:
        basis = sh_basis(order, azimuth, elevation)
        # D0 = Y / N.  The common 1/N factor is removed by the row
        # normalisation below, but it is kept explicit for clarity.
        raw = [basis[channel] / n_speakers for channel in range(n_channels)]
        weighted = [raw[channel] * weights[channel] for channel in range(n_channels)]
        norm = math.sqrt(sum(value * value for value in weighted))
        matrix.append([value / norm for value in weighted])
    return matrix


def apply_matrix(ambi: list[float], matrix: list[list[float]]) -> list[float]:
    """Decode one ambisonic frame with a speaker-by-channel matrix.

    ``matrix`` has one row per speaker and ``len(ambi)`` columns.  The raw
    output is ``matrix @ ambi``.  A decoder that uses max-rE weights or that
    has a different number of speakers can change the raw gain, so the whole
    output vector is rescaled to make ``sum(out**2) == sum(ambi**2)``.  Total
    energy must therefore be independent of the speaker count.  A silent frame
    stays silent, and inputs are not mutated.

    Energy equality must hold to within ``1e-9``.
    """
    raw = [
        sum(entry * coefficient for entry, coefficient in zip(row, ambi))
        for row in matrix
    ]

    input_energy = sum(value * value for value in ambi)
    output_energy = sum(value * value for value in raw)
    if input_energy <= 0.0 or output_energy <= 0.0:
        # A silent frame stays silent; nothing to rescale.
        return raw

    gain = math.sqrt(input_energy / output_energy)
    return [value * gain for value in raw]


def rotate(ambi: list[float], azimuth: float, elevation: float) -> list[float]:
    """Actively rotate a first-order sound field.

    The first-order direction vector ``(X, Y, Z) / sqrt(3)`` is yawed about
    ``+Z`` by ``azimuth``, then pitched about ``+Y`` so that positive
    ``elevation`` moves ``+X`` toward ``+Z``.  The omnidirectional W channel is
    invariant under rotation and must be returned unchanged; rotating a pure W
    field by any angle leaves it exactly ``[W, 0, 0, 0]``.

    ``ambi`` may contain one channel or four channels in ACN order.  Return a
    new list.  A full ``2*pi`` rotation is the identity, and first-order energy
    is conserved, to within ``1e-9``.
    """
    result = list(ambi)
    if len(result) == 1:
        # W-only field: the omnidirectional component is rotation invariant.
        return result

    # ACN order is [W, Y, Z, X]; unpack the Cartesian direction components.
    w, y, z, x = result
    cos_az = math.cos(azimuth)
    sin_az = math.sin(azimuth)

    # Yaw about +Z (counter-clockwise from +X toward +Y).
    yawed_x = cos_az * x - sin_az * y
    yawed_y = sin_az * x + cos_az * y
    yawed_z = z

    cos_el = math.cos(elevation)
    sin_el = math.sin(elevation)

    # Pitch about +Y: positive elevation moves +X toward +Z.
    pitched_x = cos_el * yawed_x + sin_el * yawed_z
    pitched_y = yawed_y
    pitched_z = -sin_el * yawed_x + cos_el * yawed_z

    # W is never modulated by a rotation.
    return [w, pitched_y, pitched_z, pitched_x]


def energy_ratio(ambi: list[float]) -> float:
    """Return the W-channel share of the order-weighted field energy.

    N3D first-order channels carry three times the energy of SN3D for the same
    direction, so convert their contribution back to SN3D energy before taking
    the ratio::

        W**2 / (W**2 + (Y**2 + Z**2 + X**2) / 3)

    For a unit plane wave this is ``0.5``; for a W-only field it is ``1.0``;
    for silence it is ``0.0``.  The result is always in ``[0, 1]`` and must be
    accurate to ``1e-9``.
    """
    w_energy = ambi[0] * ambi[0]
    first_order_energy = sum(value * value for value in ambi[1:]) / 3.0
    total = w_energy + first_order_energy
    if total <= 0.0:
        return 0.0
    return w_energy / total


def validate_basis(basis: list[list[float]], order: int) -> list[str]:
    """Validate uniformly sampled N3D basis vectors.

    ``basis`` is a non-empty sequence of samples, each a coefficient vector of
    length ``(order + 1) ** 2`` in ACN order.  For a uniform spherical sample
    set the N3D Gram matrix is the identity: every channel has mean square one
    and distinct channels have mean product zero.

    Return a list of human-readable error strings; an empty list means the
    basis is valid.  Checking channel count alone is insufficient: a basis
    with correlated Y and X channels must be reported as non-orthogonal.
    Malformed input is reported, not raised.  Tolerance is ``1e-9``.
    """
    errors: list[str] = []
    if order not in (0, 1):
        errors.append(f"order must be 0 or 1, got {order!r}")
        return errors

    n_channels = (order + 1) ** 2
    if not basis:
        errors.append("basis must contain at least one sample")
        return errors

    sample_count = len(basis)
    for index, sample in enumerate(basis):
        if len(sample) != n_channels:
            errors.append(
                f"sample {index} has {len(sample)} channels, expected {n_channels}"
            )
    if errors:
        return errors

    # Mean Gram matrix over the uniform sample set.  N3D harmonics give the
    # identity: unit mean square per channel and zero cross-channel product.
    for first in range(n_channels):
        mean_square = (
            sum(sample[first] * sample[first] for sample in basis) / sample_count
        )
        if abs(mean_square - 1.0) > _TOL:
            errors.append(
                f"channel {first} is not normalized: mean square {mean_square!r}"
            )
        for second in range(first + 1, n_channels):
            mean_product = (
                sum(
                    sample[first] * sample[second] for sample in basis
                )
                / sample_count
            )
            if abs(mean_product) > _TOL:
                errors.append(
                    f"channels {first} and {second} are not orthogonal: "
                    f"mean product {mean_product!r}"
                )
    return errors


def decode_batch(frames: list[list[float]], config: dict) -> dict:
    """Decode a batch of independent first-order Ambisonic frames.

    ``config`` contains:

    * ``"layout"``: sequence of ``(azimuth, elevation)`` speaker directions.
    * ``"order"``: ``0`` or ``1`` (defaults to ``1``).
    * ``"rotation"``: optional ``(azimuth, elevation)`` applied to every frame.

    Return a dict with keys ``"order"``, ``"layout"``, ``"matrix"``,
    ``"signals"`` and ``"energies"``.  ``"signals"`` contains one decoded
    speaker vector per input frame.  Each frame must start from its own
    original coefficients: a rotation is applied exactly once per frame, and
    frame ``i`` must never inherit the rotated state of frame ``i - 1``.
    ``"energies"`` holds the total decoded energy of each frame.  Inputs must
    not be mutated.  Results must agree to within ``1e-9``.
    """
    layout = config["layout"]
    order = config.get("order", 1)
    rotation = config.get("rotation")

    matrix = decode_matrix(len(layout), layout, order)

    signals: list[list[float]] = []
    energies: list[float] = []
    for frame in frames:
        # Every frame starts from its own untouched coefficients; the rotation
        # produces a fresh list and is never carried across frames.
        current = list(frame)
        if rotation is not None:
            current = rotate(current, rotation[0], rotation[1])
        decoded = apply_matrix(current, matrix)
        signals.append(decoded)
        energies.append(sum(value * value for value in decoded))

    return {
        "order": order,
        "layout": layout,
        "matrix": matrix,
        "signals": signals,
        "energies": energies,
    }
