"""MLC aperture generation using the DoseRAD Geant4 conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates, zoom


FloatArray = NDArray[np.float32]


def build_aperture_xy_1mm(
    mlc_left_mm: NDArray[np.floating],
    mlc_right_mm: NDArray[np.floating],
    *,
    field_size_mm: int = 400,
    leaf_thickness_mm: int = 5,
) -> FloatArray:
    """Build the simulation aperture in ``(x, y)`` array order.

    This mirrors ``geant4-dose-sim-main/code/photon_sim.py::get_aperture``.
    The field covers ``[-200, 200)`` mm at 1 mm per pixel.
    """

    left = np.asarray(mlc_left_mm, dtype=np.float64)
    right = np.asarray(mlc_right_mm, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("MLC arrays must be one-dimensional and have equal shape")
    expected_pairs = field_size_mm // leaf_thickness_mm
    if left.size != expected_pairs:
        raise ValueError(
            f"expected {expected_pairs} MLC pairs for a {field_size_mm} mm field, "
            f"got {left.size}"
        )

    aperture = np.zeros((field_size_mm, field_size_mm), dtype=np.float32)
    half = field_size_mm / 2.0
    for leaf_idx, (leaf_left, leaf_right) in enumerate(zip(left, right)):
        x0 = int(np.clip(round(float(leaf_left)) + half, 0, field_size_mm))
        x1 = int(np.clip(round(float(leaf_right)) + half, 0, field_size_mm))
        if x1 <= x0:
            continue
        y0 = leaf_idx * leaf_thickness_mm
        y1 = (leaf_idx + 1) * leaf_thickness_mm
        aperture[x0:x1, y0:y1] = 1.0
    return aperture


def to_xiao_projection_plane(
    aperture_xy_1mm: NDArray[np.floating],
    *,
    lateral_size: int = 200,
) -> FloatArray:
    """Convert the simulation aperture to Xiao's 200×200 BEV plane.

    The seemingly redundant flips/rotation intentionally reproduce the two
    reference steps exactly:

    1. Geant4 segment serialization: ``flip(aperture.T, axis=0)``;
    2. Xiao preprocessing: 0.5 linear zoom, clockwise rotation, axis-0 flip.
    """

    aperture = np.asarray(aperture_xy_1mm, dtype=np.float32)
    if aperture.shape != (400, 400):
        raise ValueError(f"expected a 400x400 aperture, got {aperture.shape}")
    serialized = np.flip(aperture.T, axis=0)
    plane = zoom(serialized, 0.5, order=1, prefilter=False)
    plane = np.flip(np.rot90(plane, 3), axis=0)
    if plane.shape != (200, 200):
        raise RuntimeError(f"unexpected projection-plane shape {plane.shape}")
    if lateral_size < 200 or lateral_size % 2:
        raise ValueError("lateral_size must be an even integer >= 200")
    if lateral_size > 200:
        padding = (lateral_size - 200) // 2
        expanded = np.zeros((lateral_size, lateral_size), dtype=np.float32)
        expanded[padding : padding + 200, padding : padding + 200] = plane
        plane = expanded
    return np.ascontiguousarray(plane, dtype=np.float32)


def project_aperture_to_bev(
    plane_2mm: NDArray[np.floating],
    *,
    depth_size: int = 352,
    lateral_size: int = 240,
    spacing_mm: float = 2.0,
    sad_mm: float = 1000.0,
    depth_before_iso_mm: float = 352.0,
) -> FloatArray:
    """Project a 2D isocenter-plane aperture through the 3D BEV grid.

    In BEV coordinates the source is at depth ``-SAD`` and the isocenter
    plane is at depth 0. This makes the projection independent of the patient
    gantry angle once the BEV basis has been constructed.
    """

    plane = np.asarray(plane_2mm, dtype=np.float32)
    if plane.shape != (lateral_size, lateral_size):
        raise ValueError(
            f"expected plane shape {(lateral_size, lateral_size)}, got {plane.shape}"
        )

    offset = -(lateral_size // 2) + 0.5
    lateral_mm = (np.arange(lateral_size, dtype=np.float32) + offset) * spacing_mm
    yy_mm, zz_mm = np.meshgrid(lateral_mm, lateral_mm, indexing="ij")

    result = np.empty(
        (depth_size, lateral_size, lateral_size), dtype=np.float32
    )
    for depth_idx in range(depth_size):
        depth_mm = -depth_before_iso_mm + depth_idx * spacing_mm
        source_to_voxel_mm = sad_mm + depth_mm
        if source_to_voxel_mm <= 0:
            result[depth_idx].fill(0.0)
            continue
        magnification = sad_mm / source_to_voxel_mm
        y_iso_mm = yy_mm * magnification
        z_iso_mm = zz_mm * magnification
        y_idx = y_iso_mm / spacing_mm - offset
        z_idx = z_iso_mm / spacing_mm - offset
        valid = (
            (y_idx >= 0.0)
            & (y_idx < lateral_size - 1)
            & (z_idx >= 0.0)
            & (z_idx < lateral_size - 1)
        )
        sampled = map_coordinates(
            plane,
            np.stack([y_idx, z_idx], axis=0),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        sampled[~valid] = 0.0
        result[depth_idx] = sampled
    return result
