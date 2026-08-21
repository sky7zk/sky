"""Global, absolute-value-preserving input and target normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NormalizationConfig:
    ct_min_hu: float = -1024.0
    ct_max_hu: float = 2000.0
    ct_scale_hu: float = 1000.0
    dose_scale_gy: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.ct_max_hu <= self.ct_min_hu:
            raise ValueError("ct_max_hu must be greater than ct_min_hu")
        if self.ct_scale_hu <= 0 or self.dose_scale_gy <= 0:
            raise ValueError("normalization scales must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "NormalizationConfig":
        return cls(
            ct_min_hu=float(values["ct_min_hu"]),
            ct_max_hu=float(values["ct_max_hu"]),
            ct_scale_hu=float(values["ct_scale_hu"]),
            dose_scale_gy=float(values["dose_scale_gy"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "NormalizationConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


def normalize_ct(
    ct_hu: NDArray[np.floating], config: NormalizationConfig
) -> NDArray[np.float32]:
    clipped = np.clip(ct_hu, config.ct_min_hu, config.ct_max_hu)
    return (clipped / config.ct_scale_hu).astype(np.float32, copy=False)


def normalize_dose(
    dose_gy: NDArray[np.floating], config: NormalizationConfig
) -> NDArray[np.float32]:
    return (np.asarray(dose_gy) / config.dose_scale_gy).astype(
        np.float32, copy=False
    )


def denormalize_dose(
    dose_normalized: NDArray[np.floating], config: NormalizationConfig
) -> NDArray[np.float32]:
    return (np.asarray(dose_normalized) * config.dose_scale_gy).astype(
        np.float32, copy=False
    )
