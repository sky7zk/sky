"""SimpleITK image loading with explicit array/physical-axis conventions."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import SimpleITK as sitk


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class ImageGeometry:
    size_xyz: tuple[int, int, int]
    spacing_xyz: NDArray[np.float64]
    origin_xyz: NDArray[np.float64]
    direction: NDArray[np.float64]

    @classmethod
    def from_sitk(cls, image: sitk.Image) -> "ImageGeometry":
        return cls(
            size_xyz=tuple(int(v) for v in image.GetSize()),
            spacing_xyz=np.asarray(image.GetSpacing(), dtype=np.float64),
            origin_xyz=np.asarray(image.GetOrigin(), dtype=np.float64),
            direction=np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3),
        )

    def is_same_as(self, other: "ImageGeometry", atol: float = 1e-6) -> bool:
        return (
            self.size_xyz == other.size_xyz
            and np.allclose(self.spacing_xyz, other.spacing_xyz, atol=atol)
            and np.allclose(self.origin_xyz, other.origin_xyz, atol=atol)
            and np.allclose(self.direction, other.direction, atol=atol)
        )


@dataclass
class Volume:
    array_xyz: FloatArray
    geometry: ImageGeometry
    reference: sitk.Image


def read_volume(path: str | Path) -> Volume:
    image = sitk.ReadImage(str(path))
    array_zyx = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    array_xyz = np.ascontiguousarray(np.transpose(array_zyx, (2, 1, 0)))
    return Volume(
        array_xyz=array_xyz,
        geometry=ImageGeometry.from_sitk(image),
        reference=image,
    )


def write_like(path: str | Path, array_xyz: NDArray[np.floating], reference: sitk.Image) -> None:
    array_zyx = np.transpose(np.asarray(array_xyz, dtype=np.float32), (2, 1, 0))
    output = sitk.GetImageFromArray(array_zyx)
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(path), useCompression=True)


def write_like_atomic(
    path: str | Path,
    array_xyz: NDArray[np.floating],
    reference: sitk.Image,
) -> None:
    """Write a compressed MHA and atomically publish the completed file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.partial.mha")
    try:
        write_like(temporary, array_xyz, reference)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
