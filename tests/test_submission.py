from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import inference


def sample_metadata() -> list[dict]:
    cp = {
        "cp_uuid": "opaque",
        "gantry_angle": -180.0,
        "mlc_left_int_mm": [-20] * 80,
        "mlc_right_int_mm": [20] * 80,
        "output_info": {
            "output_file_idx": 0,
            "idx_in_output": 0,
            "minimum_cutoff": 0.0,
        },
    }
    return [
        {
            "image_file_idx": 0,
            "anatomical_region": "thoracic",
            "beams": [
                {
                    "SAD": 1000,
                    "iso_center": [0.0, 0.0, 0.0],
                    "num_mlc_leaf_pairs": 80,
                    "control_points": [cp],
                }
            ],
        }
    ]


def test_parse_segments():
    segments = inference.parse_segments(sample_metadata())
    assert len(segments) == 1
    segment = segments[0]
    assert segment.image_index == 0
    assert segment.output_file_index == 0
    assert segment.index_in_output == 0
    assert segment.mlc_left_mm.shape == (80,)


def test_write_and_audit_outputs(tmp_path: Path):
    input_root = tmp_path / "input"
    input_dir = input_root / "images/radiation-dose-calculation-source-ct-image-1"
    input_dir.mkdir(parents=True)
    old_output = inference.OUTPUT_PATH
    old_input = inference.INPUT_PATH
    inference.OUTPUT_PATH = tmp_path
    inference.INPUT_PATH = input_root
    try:
        reference = sitk.Image([3, 4, 5], sitk.sitkFloat32)
        reference.SetSpacing((2.0, 2.0, 2.0))
        reference.SetOrigin((10.0, 20.0, 30.0))
        sitk.WriteImage(reference, str(input_dir / "ct.mha"))
        dose_xyz = np.ones((3, 4, 5), dtype=np.float32)
        predicted = {(0, 0): (dose_xyz, reference, 0, 0.0)}
        segments = inference.parse_segments(sample_metadata())
        inference.write_outputs(predicted)
        inference.audit_outputs(segments)

        real = sitk.ReadImage(
            str(tmp_path / "images/stacked-radiation-dose-map-1/output.mha")
        )
        placeholder = sitk.ReadImage(
            str(tmp_path / "images/stacked-radiation-dose-map-10/output.mha")
        )
        assert real.GetDimension() == 4
        assert real.GetSize() == (3, 4, 5, 1)
        assert real.GetSpacing()[:3] == reference.GetSpacing()
        assert placeholder.GetDimension() == 4
        assert placeholder.GetSize() == (1, 1, 1, 1)
    finally:
        inference.OUTPUT_PATH = old_output
        inference.INPUT_PATH = old_input


def test_streamed_mha_stack_preserves_frames_and_geometry(tmp_path: Path):
    reference = sitk.Image([3, 4, 5], sitk.sitkFloat32)
    reference.SetSpacing((2.0, 3.0, 4.0))
    reference.SetOrigin((10.0, 20.0, 30.0))
    reference.SetDirection((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0))
    first = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    second = first + 100.0
    path = tmp_path / "output.mha"

    writer = inference._MhaStackWriter(path, reference, frame_count=2)
    writer.write(first)
    writer.write(second)
    writer.close()

    image = sitk.ReadImage(str(path))
    array_tzyx = sitk.GetArrayFromImage(image)
    assert image.GetSize() == (3, 4, 5, 2)
    assert image.GetSpacing()[:3] == reference.GetSpacing()
    assert image.GetOrigin()[:3] == reference.GetOrigin()
    assert np.array_equal(array_tzyx[0], np.transpose(first, (2, 1, 0)))
    assert np.array_equal(array_tzyx[1], np.transpose(second, (2, 1, 0)))


def test_reject_duplicate_output_position():
    metadata = sample_metadata()
    duplicate = json.loads(json.dumps(metadata[0]["beams"][0]["control_points"][0]))
    metadata[0]["beams"][0]["control_points"].append(duplicate)
    try:
        inference.parse_segments(metadata)
    except ValueError as error:
        assert "duplicate output position" in str(error)
    else:
        raise AssertionError("duplicate output position was accepted")
