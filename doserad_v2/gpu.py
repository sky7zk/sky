"""CuPy implementation of DoseRAD BEV resampling and aperture projection."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import BEVGeometry
from .image import ImageGeometry
from .resample import bev_to_patient_affine, patient_to_bev_affine


def _cupy_modules():
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cpndi
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for GPU preprocessing. Run with the llama "
            "environment or install the CUDA-matched CuPy package."
        ) from exc
    return cp, cpndi


def _clip_gpu(array: Any, clip_min: float | None, clip_max: float | None):
    cp, _ = _cupy_modules()
    if clip_min is not None or clip_max is not None:
        lower = -cp.inf if clip_min is None else clip_min
        upper = cp.inf if clip_max is None else clip_max
        array = cp.clip(array, lower, upper)
    return array


def resample_patient_to_bev_gpu(
    array_xyz: NDArray[np.floating] | Any,
    image_geometry: ImageGeometry,
    bev_geometry: BEVGeometry,
    *,
    order: int = 3,
    cval: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
    prefilter: bool | None = None,
    return_numpy: bool = False,
):
    """Resample patient ``(x,y,z)`` data onto the BEV grid with CuPy."""

    cp, cpndi = _cupy_modules()
    matrix, offset = patient_to_bev_affine(image_geometry, bev_geometry)
    input_gpu = cp.asarray(array_xyz, dtype=cp.float32)
    output = cpndi.affine_transform(
        input_gpu,
        cp.asarray(matrix, dtype=cp.float64),
        offset=cp.asarray(offset, dtype=cp.float64),
        output_shape=bev_geometry.grid.shape,
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=(order > 1) if prefilter is None else prefilter,
    ).astype(cp.float32, copy=False)
    output = _clip_gpu(output, clip_min, clip_max)
    return cp.asnumpy(output) if return_numpy else output


def resample_bev_to_patient_gpu(
    bev_array: NDArray[np.floating] | Any,
    image_geometry: ImageGeometry,
    bev_geometry: BEVGeometry,
    *,
    order: int = 3,
    cval: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
    prefilter: bool | None = None,
    return_numpy: bool = False,
):
    """Inverse-resample BEV data onto the original patient grid with CuPy."""

    cp, cpndi = _cupy_modules()
    matrix, offset = bev_to_patient_affine(image_geometry, bev_geometry)
    input_gpu = cp.asarray(bev_array, dtype=cp.float32)
    output = cpndi.affine_transform(
        input_gpu,
        cp.asarray(matrix, dtype=cp.float64),
        offset=cp.asarray(offset, dtype=cp.float64),
        output_shape=image_geometry.size_xyz,
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=(order > 1) if prefilter is None else prefilter,
    ).astype(cp.float32, copy=False)
    output = _clip_gpu(output, clip_min, clip_max)
    return cp.asnumpy(output) if return_numpy else output


def project_aperture_to_bev_gpu(
    plane_2mm: NDArray[np.floating] | Any,
    *,
    depth_size: int = 352,
    lateral_size: int = 240,
    spacing_mm: float = 2.0,
    sad_mm: float = 1000.0,
    depth_before_iso_mm: float = 352.0,
    return_numpy: bool = False,
):
    """Project the 2D aperture to all BEV depths in one GPU operation."""

    cp, cpndi = _cupy_modules()
    plane = cp.asarray(plane_2mm, dtype=cp.float32)
    if plane.shape != (lateral_size, lateral_size):
        raise ValueError(
            f"expected plane shape {(lateral_size, lateral_size)}, got {plane.shape}"
        )

    offset = cp.float32(-(lateral_size // 2) + 0.5)
    lateral_mm = (
        cp.arange(lateral_size, dtype=cp.float32) + offset
    ) * cp.float32(spacing_mm)
    yy_mm, zz_mm = cp.meshgrid(lateral_mm, lateral_mm, indexing="ij")
    depth_mm = (
        -cp.float32(depth_before_iso_mm)
        + cp.arange(depth_size, dtype=cp.float32) * cp.float32(spacing_mm)
    )[:, None, None]
    source_to_voxel = cp.float32(sad_mm) + depth_mm
    magnification = cp.float32(sad_mm) / source_to_voxel
    y_idx = yy_mm[None] * magnification / cp.float32(spacing_mm) - offset
    z_idx = zz_mm[None] * magnification / cp.float32(spacing_mm) - offset
    valid = (
        (source_to_voxel > 0)
        & (y_idx >= 0.0)
        & (y_idx < lateral_size - 1)
        & (z_idx >= 0.0)
        & (z_idx < lateral_size - 1)
    )
    coordinates = cp.stack([y_idx, z_idx], axis=0)
    output = cpndi.map_coordinates(
        plane,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(cp.float32, copy=False)
    output = cp.where(valid, output, cp.float32(0.0))
    return cp.asnumpy(output) if return_numpy else output


class ApertureProjectorGPU:
    """Reuse the fixed BEV projection coordinates across all segments."""

    def __init__(
        self,
        *,
        depth_size: int = 352,
        lateral_size: int = 240,
        spacing_mm: float = 2.0,
        sad_mm: float = 1000.0,
        depth_before_iso_mm: float = 352.0,
    ) -> None:
        cp, _ = _cupy_modules()
        self.lateral_size = lateral_size
        offset = cp.float32(-(lateral_size // 2) + 0.5)
        lateral_mm = (
            cp.arange(lateral_size, dtype=cp.float32) + offset
        ) * cp.float32(spacing_mm)
        yy_mm, zz_mm = cp.meshgrid(lateral_mm, lateral_mm, indexing="ij")
        depth_mm = (
            -cp.float32(depth_before_iso_mm)
            + cp.arange(depth_size, dtype=cp.float32) * cp.float32(spacing_mm)
        )[:, None, None]
        source_to_voxel = cp.float32(sad_mm) + depth_mm
        magnification = cp.float32(sad_mm) / source_to_voxel
        y_idx = yy_mm[None] * magnification / cp.float32(spacing_mm) - offset
        z_idx = zz_mm[None] * magnification / cp.float32(spacing_mm) - offset
        self.valid = (
            (source_to_voxel > 0)
            & (y_idx >= 0.0)
            & (y_idx < lateral_size - 1)
            & (z_idx >= 0.0)
            & (z_idx < lateral_size - 1)
        )
        self.coordinates = cp.stack([y_idx, z_idx], axis=0)

    def __call__(self, plane_2mm: NDArray[np.floating] | Any):
        cp, cpndi = _cupy_modules()
        plane = cp.asarray(plane_2mm, dtype=cp.float32)
        expected_shape = (self.lateral_size, self.lateral_size)
        if plane.shape != expected_shape:
            raise ValueError(
                f"expected plane shape {expected_shape}, got {plane.shape}"
            )
        output = cpndi.map_coordinates(
            plane,
            self.coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(cp.float32, copy=False)
        return cp.where(self.valid, output, cp.float32(0.0))


def synchronize() -> None:
    cp, _ = _cupy_modules()
    cp.cuda.Stream.null.synchronize()


def cubic_spline_coefficients_gpu(
    array: NDArray[np.floating] | Any,
    *,
    return_numpy: bool = False,
):
    """Precompute cubic B-spline coefficients for repeated resampling.

    CT coefficients can be cached once per patient and reused for every
    control point by calling the resampling functions with ``prefilter=False``.
    """

    cp, cpndi = _cupy_modules()
    coefficients = cpndi.spline_filter(
        cp.asarray(array, dtype=cp.float32), order=3, mode="mirror"
    ).astype(cp.float32, copy=False)
    return cp.asnumpy(coefficients) if return_numpy else coefficients
