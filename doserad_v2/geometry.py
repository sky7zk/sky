"""Beam's-eye-view geometry for DoseRAD photon control points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BEVGrid:
    # Dataset coverage audit (75 patients / 40,500 control points and 150
    # sampled dose labels) showed that Xiao's 512x400x400 mm cuboid can crop
    # clinically relevant dose.  This 704x480x480 mm grid removed all sampled
    # >10% Dmax cropping while keeping the volume smaller than the more
    # conservative 384x248x248 alternative.
    shape: tuple[int, int, int] = (352, 240, 240)
    spacing_mm: tuple[float, float, float] = (2.0, 2.0, 2.0)
    depth_before_iso_mm: float = 352.0

    @property
    def offset(self) -> FloatArray:
        _, ny, nz = self.shape
        return np.asarray(
            [0.0, -(ny // 2) + 0.5, -(nz // 2) + 0.5],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class BEVGeometry:
    """World-space definition of one DoseRAD BEV cuboid."""

    gantry_angle_deg: float
    sad_mm: float
    iso_center_mm: FloatArray
    basis: FloatArray
    virtual_source_mm: FloatArray
    grid_start_mm: FloatArray
    grid: BEVGrid

    @classmethod
    def from_control_point(
        cls,
        *,
        gantry_angle_deg: float,
        sad_mm: float,
        iso_center_mm: NDArray[np.floating],
        grid: BEVGrid | None = None,
    ) -> "BEVGeometry":
        bev_grid = grid or BEVGrid()
        iso = np.asarray(iso_center_mm, dtype=np.float64)
        if iso.shape != (3,):
            raise ValueError(f"iso_center must have shape (3,), got {iso.shape}")
        if sad_mm <= 0:
            raise ValueError(f"SAD must be positive, got {sad_mm}")

        theta = np.deg2rad((270.0 + float(gantry_angle_deg)) % 360.0)
        centre_direction = np.asarray(
            [np.cos(theta), np.sin(theta), 0.0], dtype=np.float64
        )
        depth_axis = -centre_direction
        lateral_axis = np.asarray(
            [np.sin(theta), -np.cos(theta), 0.0], dtype=np.float64
        )
        si_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        basis = np.stack([depth_axis, lateral_axis, si_axis], axis=0)

        virtual_source = iso - sad_mm * depth_axis
        grid_start = iso - bev_grid.depth_before_iso_mm * depth_axis
        return cls(
            gantry_angle_deg=float(gantry_angle_deg),
            sad_mm=float(sad_mm),
            iso_center_mm=iso,
            basis=basis,
            virtual_source_mm=virtual_source,
            grid_start_mm=grid_start,
            grid=bev_grid,
        )

    @property
    def depth_axis(self) -> FloatArray:
        return self.basis[0]

    @property
    def lateral_axis(self) -> FloatArray:
        return self.basis[1]

    @property
    def si_axis(self) -> FloatArray:
        return self.basis[2]

    def bev_index_to_world(self, index_dls: NDArray[np.floating]) -> FloatArray:
        """Map BEV ``(depth, lateral, SI)`` indices to world millimetres."""

        index = np.asarray(index_dls, dtype=np.float64)
        local_mm = (index + self.grid.offset) * np.asarray(
            self.grid.spacing_mm, dtype=np.float64
        )
        return self.grid_start_mm + self.basis.T @ local_mm

    def world_to_bev_index(self, point_mm: NDArray[np.floating]) -> FloatArray:
        """Map a world-space point to continuous BEV indices."""

        point = np.asarray(point_mm, dtype=np.float64)
        local_mm = self.basis @ (point - self.grid_start_mm)
        return local_mm / np.asarray(self.grid.spacing_mm) - self.grid.offset

    def validate(self, atol: float = 1e-7) -> None:
        identity = self.basis @ self.basis.T
        if not np.allclose(identity, np.eye(3), atol=atol):
            raise ValueError(f"BEV basis is not orthonormal:\n{identity}")
        if np.linalg.det(self.basis) <= 0:
            raise ValueError("BEV basis is not right-handed")
        source_distance = np.linalg.norm(
            self.iso_center_mm - self.virtual_source_mm
        )
        if not np.isclose(source_distance, self.sad_mm, atol=atol):
            raise ValueError(
                f"source-isocenter distance {source_distance} != SAD {self.sad_mm}"
            )
