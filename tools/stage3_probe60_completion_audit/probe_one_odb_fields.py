from __future__ import print_function

import json
from pathlib import Path

from odbAccess import openOdb


ODB_PATH = Path(r"E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_true_variable_N_probe60_v01\N24\N24_A07_regular_jump_coprime\J2D_N24_N24_A07_regular_jump_coprime.odb")
OUT_PATH = Path(r"E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\N24_A07_odb_field_probe.json")


def sample_field(field):
    values = field.values
    sample = values[0] if values else None
    data = getattr(sample, "data", None) if sample else None
    if hasattr(data, "__iter__") and not isinstance(data, str):
        sample_data = [float(x) for x in data]
    elif data is None:
        sample_data = None
    else:
        sample_data = float(data)
    sample_mises = getattr(sample, "mises", None) if sample else None
    sample_max_principal = getattr(sample, "maxPrincipal", None) if sample else None
    return {
        "componentLabels": list(getattr(field, "componentLabels", []) or []),
        "validInvariants": [str(x) for x in (getattr(field, "validInvariants", []) or [])],
        "value_count": len(values),
        "sample_position": str(getattr(sample, "position", "")) if sample else None,
        "sample_data": sample_data,
        "sample_mises": float(sample_mises) if sample_mises is not None else None,
        "sample_maxPrincipal": float(sample_max_principal) if sample_max_principal is not None else None,
    }


def main():
    odb = openOdb(path=str(ODB_PATH), readOnly=True)
    try:
        step = odb.steps["step_final_cooling"]
        frame = step.frames[-1]
        payload = {
            "odb_path": str(ODB_PATH),
            "step": step.name,
            "frame_description": frame.description,
            "fields": {},
        }
        for key in ["U", "PEEQ", "S", "NT11"]:
            payload["fields"][key] = sample_field(frame.fieldOutputs[key])
        OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"status": "ok", "out": str(OUT_PATH)}, indent=2))
    finally:
        odb.close()


if __name__ == "__main__":
    main()
