"""Reference CPU resampling between patient and BEV coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import affine_transform

from .geometry import BEVGeometry
from .image import ImageGeometry


FloatArray = NDArray[np.float32]


def patient_to_bev_affine(
    image_geometry: ImageGeometry,
    bev_geometry: BEVGeometry,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return SciPy output-BEV-index → input-patient-index affine."""

    spacing_patient = np.diag(image_geometry.spacing_xyz)
    inv_spacing_patient = np.diag(1.0 / image_geometry.spacing_xyz)
    spacing_bev = np.diag(np.asarray(bev_geometry.grid.spacing_mm))
    inv_direction = np.linalg.inv(image_geometry.direction)

    matrix = (
        inv_spacing_patient
        @ inv_direction
        @ bev_geometry.basis.T
        @ spacing_bev
    )
    offset = inv_spacing_patient @ inv_direction @ (
        bev_geometry.grid_start_mm
        + bev_geometry.basis.T @ (spacing_bev @ bev_geometry.grid.offset)
        - image_geometry.origin_xyz
    )
    return matrix, offset


def bev_to_patient_affine(
    image_geometry: ImageGeometry,
    bev_geometry: BEVGeometry,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return SciPy output-patient-index → input-BEV-index affine."""

    spacing_patient = np.diag(image_geometry.spacing_xyz)
    inv_spacing_bev = np.diag(1.0 / np.asarray(bev_geometry.grid.spacing_mm))
    matrix = (
        inv_spacing_bev
        @ bev_geometry.basis
        @ image_geometry.direction
        @ spacing_patient
    )
    offset = (
        inv_spacing_bev
        @ bev_geometry.basis
        @ (image_geometry.origin_xyz - bev_geometry.grid_start_mm)
        - bev_geometry.grid.offset
    )
    return matrix, offset


def resample_patient_to_bev(
    array_xyz: NDArray[np.floating],
    image_geometry: ImageGeometry,
    bev_geometry: BEVGeometry,
    *,
    order: int = 3,
    cval: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> FloatArray:
    matrix, offset = patient_to_bev_affine(image_geometry, bev_geometry)
    output = affine_transform(
        np.asarray(array_xyz, dtype=np.float32),
        matrix,
        offset=offset,
        output_shape=bev_geometry.grid.shape,
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=order > 1,
    ).astype(np.float32, copy=False)
    if clip_min is not None or clip_max is not None:
        output = np.clip(output, clip_min, clip_max)
    return output


def resample_bev_to_patient(
    bev_array: NDArray[np.floating],
    image_geometry: ImageGeometry,
    bev_geometry: BEVGeometry,
    *,
    order: int = 3,
    cval: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> FloatArray:
    matrix, offset = bev_to_patient_affine(image_geometry, bev_geometry)
    output = affine_transform(
        np.asarray(bev_array, dtype=np.float32),
        matrix,
        offset=offset,
        output_shape=image_geometry.size_xyz,
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=order > 1,
    ).astype(np.float32, copy=False)
    if clip_min is not None or clip_max is not None:
        output = np.clip(output, clip_min, clip_max)
    return output

