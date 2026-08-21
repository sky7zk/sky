"""Convert one public training patient into Grand Challenge test input format."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import SimpleITK as sitk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-dir", required=True, type=Path)
    parser.add_argument("--output", default=Path("test/input"), type=Path)
    parser.add_argument("--control-points", default=4, type=int)
    args = parser.parse_args()
    if args.control_points <= 0:
        raise ValueError("control-points must be positive")

    patient_id = args.patient_dir.name
    plan_path = args.patient_dir / f"{patient_id}.json"
    ct_path = args.patient_dir / "image" / "ct.mha"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    selected_beams = []
    output_index = 0
    for source_beam in plan["beams"]:
        selected_control_points = []
        for source_cp in source_beam["control_points"]:
            if output_index >= args.control_points:
                break
            cp = copy.deepcopy(source_cp)
            cp.pop("cp_idx", None)
            cp["cp_uuid"] = f"local-{output_index}"
            cp["output_info"] = {
                "output_file_idx": 0,
                "idx_in_output": output_index,
                "minimum_cutoff": 0.0,
            }
            selected_control_points.append(cp)
            output_index += 1
        if selected_control_points:
            beam = copy.deepcopy(source_beam)
            beam.pop("beam_idx", None)
            beam["control_points"] = selected_control_points
            selected_beams.append(beam)
        if output_index >= args.control_points:
            break
    if output_index != args.control_points:
        raise ValueError(f"patient contains only {output_index} control points")

    if args.output.exists():
        shutil.rmtree(args.output)
    images = args.output / "images"
    images.mkdir(parents=True)
    for slot in range(1, 11):
        directory = images / f"radiation-dose-calculation-source-ct-image-{slot}"
        directory.mkdir()
        if slot == 1:
            shutil.copy2(ct_path, directory / "ct.mha")
        else:
            sitk.WriteImage(sitk.Image([1, 1, 1], sitk.sitkFloat32), str(directory / "placeholder.mha"))

    metadata = [
        {
            "image_file_idx": 0,
            "anatomical_region": "thoracic" if "TH" in patient_id else "abdominal",
            "beams": selected_beams,
        }
    ]
    (args.output / "stacked-photon-beam-level-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} with {output_index} control points")


if __name__ == "__main__":
    main()

