from __future__ import print_function

import json
from pathlib import Path

from odbAccess import openOdb


ODB_PATH = Path(r"E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_true_variable_N_probe60_v01\N24\N24_A07_regular_jump_coprime\J2D_N24_N24_A07_regular_jump_coprime.odb")
OUT_PATH = Path(r"E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\N24_A07_odb_structure_probe.json")


def main():
    odb = openOdb(path=str(ODB_PATH), readOnly=True)
    try:
        steps = []
        for step_name, step in odb.steps.items():
            frames = step.frames
            field_keys = sorted(list(frames[-1].fieldOutputs.keys())) if frames else []
            steps.append(
                {
                    "name": step_name,
                    "frame_count": len(frames),
                    "last_frame_value": frames[-1].frameValue if frames else None,
                    "last_frame_description": frames[-1].description if frames else None,
                    "last_frame_field_outputs": field_keys,
                }
            )
        root = odb.rootAssembly
        payload = {
            "odb_path": str(ODB_PATH),
            "steps": steps,
            "root_instance_names": sorted(list(root.instances.keys())),
            "root_node_set_names": sorted(list(root.nodeSets.keys())),
            "root_element_set_names": sorted(list(root.elementSets.keys())),
            "root_surface_names": sorted(list(root.surfaces.keys())),
        }
        OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"status": "ok", "out": str(OUT_PATH)}, indent=2))
    finally:
        odb.close()


if __name__ == "__main__":
    main()
