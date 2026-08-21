"""DoseRAD2026 Photon-on-CT Grand Challenge inference adapter.

The platform mounts up to ten CT images and one stacked photon metadata JSON
under /input. This module predicts every requested control point, reconstructs
the dose on the original CT grid, and writes ten genuine 4-D MHA stacks under
/output/images.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import glob
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import cupy as cp
import numpy as np
import SimpleITK as sitk
import torch

from doserad_v2.aperture import build_aperture_xy_1mm, to_xiao_projection_plane
from doserad_v2.geometry import BEVGeometry
from doserad_v2.gpu import (
    ApertureProjectorGPU,
    cubic_spline_coefficients_gpu,
    resample_bev_to_patient_gpu,
    resample_patient_to_bev_gpu,
)
from doserad_v2.image import Volume, read_volume
from doserad_v2.inference import load_inference_weights
from doserad_v2.model import CNNConvLSTM
from doserad_v2.normalization import NormalizationConfig


INPUT_PATH = Path(os.environ.get("GRAND_CHALLENGE_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("GRAND_CHALLENGE_OUTPUT", "/output"))
MODEL_PATH = Path(os.environ.get("GRAND_CHALLENGE_MODEL", "/opt/ml/model"))
NUM_SLOTS = 10
CT_SOCKET = "radiation-dose-calculation-source-ct-image"
METADATA_SOCKET = "stacked-photon-beam-level-metadata"


@dataclass(frozen=True)
class Segment:
    image_index: int
    beam_index: int
    cp_index: int
    gantry_angle: float
    sad_mm: float
    iso_center_mm: np.ndarray
    mlc_left_mm: np.ndarray
    mlc_right_mm: np.ndarray
    output_file_index: int
    index_in_output: int
    minimum_cutoff: float


@dataclass
class RuntimeModel:
    model: CNNConvLSTM
    normalization: NormalizationConfig
    device: torch.device
    batch_size: int
    weight_variant: str


def _one_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if len(files) != 1:
        raise ValueError(f"expected exactly one {pattern} in {directory}, found {len(files)}")
    return files[0]


def _find_resource(name: str) -> Path:
    matches = sorted(MODEL_PATH.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} below {MODEL_PATH}, found {len(matches)}")
    return matches[0]


def load_runtime_model() -> RuntimeModel:
    if not torch.cuda.is_available():
        raise RuntimeError("DoseRAD photon inference requires a CUDA GPU")
    checkpoint_path = _find_resource("best_full_ct_beam_mae.pth")
    normalization_path = _find_resource("normalization.json")
    normalization = NormalizationConfig.from_json(normalization_path)
    device = torch.device("cuda")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CNNConvLSTM(output_init="small").to(device).eval()
    variant = load_inference_weights(model, state)
    del state
    batch_size = int(os.environ.get("DOSERAD_BATCH_SIZE", "2"))
    if batch_size <= 0:
        raise ValueError("DOSERAD_BATCH_SIZE must be positive")
    print(
        f"Loaded {checkpoint_path.name} ({variant}) on {torch.cuda.get_device_name(0)}; "
        f"batch_size={batch_size}",
        flush=True,
    )
    return RuntimeModel(model, normalization, device, batch_size, variant)


def load_metadata(path: Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        path = _one_file(INPUT_PATH, f"{METADATA_SOCKET}*.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("stacked photon metadata must be a non-empty JSON array")
    return data


def parse_segments(metadata: list[dict[str, Any]]) -> list[Segment]:
    """Flatten metadata and assign local indices when private IDs are absent."""

    result: list[Segment] = []
    seen_positions: set[tuple[int, int]] = set()
    for image in metadata:
        image_index = int(image["image_file_idx"])
        if not 0 <= image_index < NUM_SLOTS:
            raise ValueError(f"image_file_idx out of range: {image_index}")
        for beam_index, beam in enumerate(image["beams"]):
            sad_mm = float(beam["SAD"])
            iso = np.asarray(beam["iso_center"], dtype=np.float64)
            pair_count = int(beam["num_mlc_leaf_pairs"])
            for cp_index, control_point in enumerate(beam["control_points"]):
                left = np.asarray(control_point["mlc_left_int_mm"], dtype=np.float64)
                right = np.asarray(control_point["mlc_right_int_mm"], dtype=np.float64)
                if left.shape != (pair_count,) or right.shape != (pair_count,):
                    raise ValueError(
                        f"image {image_index} beam {beam_index} cp {cp_index}: "
                        f"expected {pair_count} MLC pairs"
                    )
                info = control_point["output_info"]
                output_file_index = int(info["output_file_idx"])
                index_in_output = int(info["idx_in_output"])
                position = (output_file_index, index_in_output)
                if not 0 <= output_file_index < NUM_SLOTS or index_in_output < 0:
                    raise ValueError(f"invalid output position {position}")
                if position in seen_positions:
                    raise ValueError(f"duplicate output position {position}")
                seen_positions.add(position)
                result.append(
                    Segment(
                        image_index=image_index,
                        beam_index=beam_index,
                        cp_index=cp_index,
                        gantry_angle=float(control_point["gantry_angle"]),
                        sad_mm=sad_mm,
                        iso_center_mm=iso,
                        mlc_left_mm=left,
                        mlc_right_mm=right,
                        output_file_index=output_file_index,
                        index_in_output=index_in_output,
                        minimum_cutoff=float(info["minimum_cutoff"]),
                    )
                )
    if not result:
        raise ValueError("metadata contains no photon control points")
    for output_index in range(NUM_SLOTS):
        indices = sorted(s.index_in_output for s in result if s.output_file_index == output_index)
        if indices and indices != list(range(len(indices))):
            raise ValueError(f"output slot {output_index} indices are not contiguous: {indices}")
    return result


def _input_image_path(image_index: int) -> Path:
    directory = INPUT_PATH / "images" / f"{CT_SOCKET}-{image_index + 1}"
    return _one_file(directory, "*.mha")


def _projector(cache: dict[tuple[float, ...], ApertureProjectorGPU], sad: float, geometry: BEVGeometry):
    key = (
        float(sad),
        float(geometry.grid.depth_before_iso_mm),
        *map(float, geometry.grid.shape),
        *map(float, geometry.grid.spacing_mm),
    )
    if key not in cache:
        cache[key] = ApertureProjectorGPU(
            sad_mm=sad,
            depth_size=geometry.grid.shape[0],
            lateral_size=geometry.grid.shape[1],
            spacing_mm=geometry.grid.spacing_mm[0],
            depth_before_iso_mm=geometry.grid.depth_before_iso_mm,
        )
    return cache[key]


def _build_inputs(
    segment: Segment,
    volume: Volume,
    coefficients,
    normalization: NormalizationConfig,
    projector_cache: dict[tuple[float, ...], ApertureProjectorGPU],
):
    geometry = BEVGeometry.from_control_point(
        gantry_angle_deg=segment.gantry_angle,
        sad_mm=segment.sad_mm,
        iso_center_mm=segment.iso_center_mm,
    )
    ct_bev = resample_patient_to_bev_gpu(
        coefficients,
        volume.geometry,
        geometry,
        order=3,
        cval=-1024.0,
        clip_min=-1024.0,
        clip_max=float(volume.array_xyz.max()),
        prefilter=False,
    )
    aperture = build_aperture_xy_1mm(segment.mlc_left_mm, segment.mlc_right_mm)
    plane = to_xiao_projection_plane(aperture, lateral_size=geometry.grid.shape[1])
    plane = np.rint(np.clip(plane, 0.0, 1.0) * 255.0).astype(np.float32) / np.float32(255.0)
    projection = _projector(projector_cache, segment.sad_mm, geometry)(plane)
    ct_normalized = cp.clip(
        ct_bev, normalization.ct_min_hu, normalization.ct_max_hu
    ) / cp.float32(normalization.ct_scale_hu)
    return (
        torch.from_dlpack(ct_normalized.astype(cp.float32, copy=False)),
        torch.from_dlpack(projection.astype(cp.float32, copy=False)),
        geometry,
    )


def _batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def predict(runtime: RuntimeModel, segments: list[Segment]):
    """Return output-position keyed patient-space dose arrays and references."""

    predicted: dict[tuple[int, int], tuple[np.ndarray, sitk.Image, int, float]] = {}
    projector_cache: dict[tuple[float, ...], ApertureProjectorGPU] = {}
    for image_index in sorted({s.image_index for s in segments}):
        image_segments = [s for s in segments if s.image_index == image_index]
        volume = read_volume(_input_image_path(image_index))
        if volume.reference.GetDimension() != 3:
            raise ValueError(f"input slot {image_index} is not a 3-D CT image")
        coefficients = cubic_spline_coefficients_gpu(volume.array_xyz)
        for segment_batch in _batched(image_segments, runtime.batch_size):
            inputs = [
                _build_inputs(s, volume, coefficients, runtime.normalization, projector_cache)
                for s in segment_batch
            ]
            ct = torch.stack([item[0] for item in inputs])
            projection = torch.stack([item[1] for item in inputs])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                bev_predictions = runtime.model(ct, projection)
            for index, (segment, (_, _, geometry)) in enumerate(zip(segment_batch, inputs)):
                bev = cp.maximum(
                    cp.from_dlpack(bev_predictions[index].float())
                    * cp.float32(runtime.normalization.dose_scale_gy),
                    cp.float32(0.0),
                )
                patient = resample_bev_to_patient_gpu(
                    bev,
                    volume.geometry,
                    geometry,
                    order=3,
                    cval=0.0,
                    clip_min=0.0,
                    prefilter=True,
                    return_numpy=True,
                )
                if not np.isfinite(patient).all():
                    raise FloatingPointError(f"non-finite prediction at {segment}")
                # The challenge explicitly requires all positive values at or
                # below the per-dose-map cutoff to be zero.
                patient[patient <= segment.minimum_cutoff] = 0.0
                predicted[(segment.output_file_index, segment.index_in_output)] = (
                    patient,
                    volume.reference,
                    segment.image_index,
                    segment.minimum_cutoff,
                )
            del ct, projection, bev_predictions, inputs
        del coefficients
        cp.get_default_memory_pool().free_all_blocks()
    return predicted


def _as_frame(array_xyz: np.ndarray, reference: sitk.Image) -> sitk.Image:
    array_zyx = np.transpose(np.asarray(array_xyz, dtype=np.float32), (2, 1, 0))
    frame = sitk.GetImageFromArray(array_zyx)
    frame.CopyInformation(reference)
    return frame


def _placeholder_4d() -> sitk.Image:
    frame = sitk.Image([1, 1, 1], sitk.sitkFloat32)
    return sitk.JoinSeries([frame])


def write_outputs(predicted) -> None:
    output_images = OUTPUT_PATH / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    for output_index in range(NUM_SLOTS):
        directory = output_images / f"stacked-radiation-dose-map-{output_index + 1}"
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        rows = sorted(
            ((position[1], value) for position, value in predicted.items() if position[0] == output_index),
            key=lambda item: item[0],
        )
        if not rows:
            image = _placeholder_4d()
        else:
            expected = list(range(len(rows)))
            actual = [idx for idx, _ in rows]
            if actual != expected:
                raise ValueError(f"output slot {output_index} indices {actual} != {expected}")
            image_indices = {value[2] for _, value in rows}
            if len(image_indices) != 1:
                raise ValueError(f"output slot {output_index} mixes CT images: {image_indices}")
            frames = [_as_frame(value[0], value[1]) for _, value in rows]
            image = sitk.JoinSeries(frames)
        sitk.WriteImage(image, str(directory / "output.mha"), useCompression=False)


def audit_outputs(segments: list[Segment]) -> None:
    expected_counts = {
        output_index: sum(s.output_file_index == output_index for s in segments)
        for output_index in range(NUM_SLOTS)
    }
    for output_index in range(NUM_SLOTS):
        path = _one_file(
            OUTPUT_PATH / "images" / f"stacked-radiation-dose-map-{output_index + 1}",
            "*.mha",
        )
        image = sitk.ReadImage(str(path))
        if image.GetDimension() != 4:
            raise ValueError(f"{path} is {image.GetDimension()}-D, expected 4-D")
        count = expected_counts[output_index]
        expected_frames = count if count else 1
        if image.GetSize()[3] != expected_frames:
            raise ValueError(f"{path} has {image.GetSize()[3]} frames, expected {expected_frames}")
        array = sitk.GetArrayFromImage(image)
        if not np.isfinite(array).all() or float(array.min()) < 0.0:
            raise ValueError(f"{path} contains invalid dose values")
        slot_segments = sorted(
            (s for s in segments if s.output_file_index == output_index),
            key=lambda s: s.index_in_output,
        )
        if not slot_segments:
            if image.GetSize() != (1, 1, 1, 1) or np.any(array):
                raise ValueError(f"{path} is not a valid zero placeholder")
            continue

        image_indices = {s.image_index for s in slot_segments}
        if len(image_indices) != 1:
            raise ValueError(f"output slot {output_index} mixes CT images: {image_indices}")
        reference = sitk.ReadImage(str(_input_image_path(next(iter(image_indices)))))
        if image.GetSize()[:3] != reference.GetSize():
            raise ValueError(f"{path} size does not match its source CT")
        if not np.allclose(image.GetSpacing()[:3], reference.GetSpacing(), atol=1e-6):
            raise ValueError(f"{path} spacing does not match its source CT")
        if not np.allclose(image.GetOrigin()[:3], reference.GetOrigin(), atol=1e-6):
            raise ValueError(f"{path} origin does not match its source CT")
        direction4 = np.asarray(image.GetDirection(), dtype=np.float64).reshape(4, 4)
        direction3 = np.asarray(reference.GetDirection(), dtype=np.float64).reshape(3, 3)
        if not np.allclose(direction4[:3, :3], direction3, atol=1e-6):
            raise ValueError(f"{path} direction does not match its source CT")
        for frame, segment in zip(array, slot_segments):
            invalid = (frame > 0.0) & (frame <= segment.minimum_cutoff)
            if np.any(invalid):
                raise ValueError(
                    f"{path} frame {segment.index_in_output} contains positive "
                    f"values <= cutoff {segment.minimum_cutoff}"
                )


def run(runtime: RuntimeModel) -> None:
    started = time.perf_counter()
    metadata = load_metadata()
    segments = parse_segments(metadata)
    print(f"Invoking Photon-CT inference for {len(metadata)} images, {len(segments)} control points")
    predicted = predict(runtime, segments)
    write_outputs(predicted)
    audit_outputs(segments)
    elapsed = time.perf_counter() - started
    print(f"DoseRAD invoke complete: {len(segments)} control points in {elapsed:.3f}s", flush=True)
