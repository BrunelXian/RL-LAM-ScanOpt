"""Abaqus noGUI generator for Stage 3 true variable-N probe60 cases.

Default behavior is a normal-Python-compatible dry run. Generate mode is
fail-closed until the user explicitly confirms heat-load mapping. Even then the
default generate target is one pilot case only.
"""

from __future__ import annotations

import csv
import json
import os
import re
import traceback
from pathlib import Path


# CONFIG ---------------------------------------------------------------------
MODE = "generate"
ALLOW_GENERIC_EXPORT_WITHOUT_REORDER = False
ALLOW_USER_CONFIRMED_HEAT_MAPPING = True
ONLY_GENERATE_ONE_PILOT_CASE = False
PILOT_N = 12
PILOT_STRATEGY = "N12_A01_raster_left_to_right"
TARGET_N_VALUES = [16, 24, 40]
SKIP_N_VALUES = [12]
HEAT_MAPPING_CONFIG_PATH = r"E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_heat_mapping_config_CONFIRMED_FULL60.json"

ALLOWED_MODES = ("dry_run", "generate")


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
CASE_ROOT = PROJECT_ROOT / "cae_model" / "stage3_true_variable_N_probe60_v01"
MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_manual_probe60_handoff"
    / "variable_N_probe60_case_manifest_FIXED.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_cae_probe60_generation"
HEAT_MAPPING_CONFIG = Path(HEAT_MAPPING_CONFIG_PATH)
BASE_MESH_AUDIT_SUMMARY = OUTPUT_DIR / "base_mesh_audit_N16_N24_N40_summary.json"

BASES = {
    12: PROJECT_ROOT / "cae_model" / "12track_full" / "sanity_base" / "12track_sanity_base.cae",
    16: PROJECT_ROOT / "cae_model" / "16track_full" / "sanity_base" / "16track_sanity_base.cae",
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.cae",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.cae",
}

BASE_JNLS = {
    12: PROJECT_ROOT / "cae_model" / "12track_full" / "sanity_base" / "12track_sanity_base.jnl",
    16: PROJECT_ROOT / "cae_model" / "16track_full" / "sanity_base" / "16track_sanity_base.jnl",
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.jnl",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.jnl",
}

EXPECTED_NS = (12, 16, 24, 40)
BAD_SCHEMA_TOKENS = ("N12N12_", "N16N16_", "N24N24_", "N40N40_")


class Probe60GenerationError(RuntimeError):
    pass


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_scan_order(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_scan_order(n: int, order: object) -> None:
    if not isinstance(order, list):
        raise Probe60GenerationError("scan_order is not a list")
    if len(order) != n:
        raise Probe60GenerationError("scan_order length {} != N {}".format(len(order), n))
    if not all(isinstance(v, int) for v in order):
        raise Probe60GenerationError("scan_order contains non-integer entries")
    if sorted(order) != list(range(n)):
        raise Probe60GenerationError("scan_order is not a permutation of 0..{}".format(n - 1))


def load_heat_mapping_config() -> dict:
    if not HEAT_MAPPING_CONFIG.exists():
        raise Probe60GenerationError("missing heat mapping config: {}".format(HEAT_MAPPING_CONFIG))
    config = json.loads(HEAT_MAPPING_CONFIG.read_text(encoding="utf-8"))
    required = (
        "scope",
        "track_set_pattern",
        "scan_step_pattern",
        "cool_step_pattern",
        "load_pattern",
        "template_scan_step",
        "template_cool_step",
        "template_load",
        "load_type",
        "body_heat_flux_magnitude",
        "scan_duration_seconds",
        "cool_duration_seconds",
        "final_cooling_step_name",
        "final_cooling_duration_seconds",
        "final_cooling_initial_increment",
        "final_cooling_max_increment_size",
        "append_final_cooling_step",
        "allow_generate_when_confirmed",
        "allow_full60_generation",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise Probe60GenerationError("heat mapping config missing keys: {}".format(missing))
    if config.get("base_model_type") != "single_scan_step_sanity_template":
        raise Probe60GenerationError("unexpected base_model_type in heat mapping config")
    if config.get("allow_generate_when_confirmed") is not True:
        raise Probe60GenerationError("allow_generate_when_confirmed must be true for confirmed generation")
    scope = config.get("scope")
    if scope == "pilot_only":
        if int(config.get("pilot_n")) != PILOT_N:
            raise Probe60GenerationError("confirmed config pilot_n does not match PILOT_N")
        if config.get("pilot_strategy") != PILOT_STRATEGY:
            raise Probe60GenerationError("confirmed config pilot_strategy does not match PILOT_STRATEGY")
        if config.get("allow_full60_generation") is not False:
            raise Probe60GenerationError("allow_full60_generation must be false for pilot config")
    elif scope == "full60_generation":
        if ONLY_GENERATE_ONE_PILOT_CASE:
            raise Probe60GenerationError("full60 config cannot run while ONLY_GENERATE_ONE_PILOT_CASE is true")
        if config.get("allow_full60_generation") is not True:
            raise Probe60GenerationError("allow_full60_generation must be true for full60 config")
        if config.get("expected_case_count") != 60:
            raise Probe60GenerationError("full60 config expected_case_count must be 60")
        if config.get("cases_per_n") != 15:
            raise Probe60GenerationError("full60 config cases_per_n must be 15")
        if list(config.get("n_values", [])) != [12, 16, 24, 40]:
            raise Probe60GenerationError("full60 config n_values must be [12, 16, 24, 40]")
    else:
        raise Probe60GenerationError("unsupported confirmed config scope: {}".format(scope))
    if not config.get("append_final_cooling_step"):
        raise Probe60GenerationError("append_final_cooling_step must be true")
    if config.get("load_type") != "BodyHeatFlux":
        raise Probe60GenerationError("confirmed config load_type must be BodyHeatFlux")
    for key in (
        "body_heat_flux_magnitude",
        "scan_duration_seconds",
        "cool_duration_seconds",
        "final_cooling_duration_seconds",
        "final_cooling_initial_increment",
        "final_cooling_max_increment_size",
    ):
        try:
            float(config[key])
        except (TypeError, ValueError):
            raise Probe60GenerationError("confirmed config {} is not numeric".format(key))
    if float(config["final_cooling_duration_seconds"]) != 1200.0:
        raise Probe60GenerationError("final cooling duration must be 1200.0 seconds")
    if float(config["final_cooling_initial_increment"]) != 0.01:
        raise Probe60GenerationError("final cooling initial increment must be 0.01")
    if float(config["final_cooling_max_increment_size"]) != 60.0:
        raise Probe60GenerationError("final cooling max increment size must be 60.0")
    return config


def validate_case_paths(row: dict[str, str], n: int) -> None:
    n_token = "N{}".format(n)
    case_dir = Path(row["case_dir"])
    expected_fragment = "\\{}\\{}_".format(n_token, n_token)
    all_paths = [
        str(case_dir),
        row["scan_order_json"],
        row["expected_cae"],
        row["expected_inp"],
        row["expected_jnl"],
    ]
    if expected_fragment not in str(case_dir):
        raise Probe60GenerationError("case_dir missing schema fragment {}".format(expected_fragment))
    for token in BAD_SCHEMA_TOKENS:
        if any(token in p for p in all_paths):
            raise Probe60GenerationError("bad concatenated path schema token found: {}".format(token))
    if case_dir.parent != CASE_ROOT / n_token:
        raise Probe60GenerationError("case_dir is not under correct N folder")
    for key in ("expected_cae", "expected_inp", "expected_jnl", "scan_order_json"):
        if Path(row[key]).parent != case_dir:
            raise Probe60GenerationError("{} is not inside case_dir".format(key))


def infer_heat_mapping_status_from_jnl() -> tuple[str, dict[str, dict[str, int]]]:
    evidence: dict[str, dict[str, int]] = {}
    for n, jnl in BASE_JNLS.items():
        if not jnl.exists():
            return "INSUFFICIENT_JNL_EVIDENCE", evidence
        text = jnl.read_text(errors="replace")
        heat_sets = set(re.findall(r"set_body_heat_\d+", text))
        scan_steps = set(re.findall(r"step_scan_\d+", text))
        cool_steps = set(re.findall(r"step_cool_\d+", text))
        body_heat_loads = set(re.findall(r"load_body_hflux_\d+", text))
        evidence[str(n)] = {
            "heat_sets": len(heat_sets),
            "scan_steps": len(scan_steps),
            "cool_steps": len(cool_steps),
            "body_heat_loads": len(body_heat_loads),
        }
        if len(heat_sets) != n:
            return "INSUFFICIENT_JNL_EVIDENCE", evidence
    return "USER_CONFIRMED_MAPPING_REQUIRED", evidence


def validate_all_inputs() -> tuple[list[dict[str, object]], str, dict[str, dict[str, int]], dict]:
    rows = load_manifest()
    if len(rows) != 60:
        raise Probe60GenerationError("manifest row count {} != 60".format(len(rows)))

    config = load_heat_mapping_config()
    for n, base in BASES.items():
        if not base.exists():
            raise Probe60GenerationError("missing base CAE for N{}: {}".format(n, base))
        if not BASE_JNLS[n].exists():
            raise Probe60GenerationError("missing base JNL for N{}: {}".format(n, BASE_JNLS[n]))

    checked_rows: list[dict[str, object]] = []
    for row in rows:
        n = int(row["n"])
        if n not in EXPECTED_NS:
            raise Probe60GenerationError("unexpected N {}".format(n))
        validate_case_paths(row, n)
        scan_data = load_scan_order(Path(row["scan_order_json"]))
        validate_scan_order(n, scan_data.get("scan_order"))
        checked_rows.append(
            {
                "n": n,
                "strategy_name": row["strategy_name"],
                "job_name": row["job_name"],
                "case_dir": row["case_dir"],
                "scan_order_json": row["scan_order_json"],
                "expected_cae": row["expected_cae"],
                "expected_inp": row["expected_inp"],
                "expected_jnl": row["expected_jnl"],
                "scan_order": scan_data["scan_order"],
            }
        )

    mapping_status, evidence = infer_heat_mapping_status_from_jnl()
    return checked_rows, mapping_status, evidence, config


def select_generate_rows(rows: list[dict[str, object]], mapping_config: dict | None = None) -> list[dict[str, object]]:
    if ONLY_GENERATE_ONE_PILOT_CASE:
        selected = [
            row
            for row in rows
            if int(row["n"]) == PILOT_N and row["strategy_name"] == PILOT_STRATEGY
        ]
        if len(selected) != 1:
            raise Probe60GenerationError("pilot case not found uniquely")
        return selected
    if mapping_config is None or mapping_config.get("scope") != "full60_generation":
        raise Probe60GenerationError("full generation requires full60_generation config")
    if mapping_config.get("allow_full60_generation") is not True:
        raise Probe60GenerationError("full generation requires allow_full60_generation true")
    target_ns = [int(n) for n in TARGET_N_VALUES]
    skip_ns = [int(n) for n in SKIP_N_VALUES]
    if target_ns:
        selected = [row for row in rows if int(row["n"]) in target_ns]
        if any(int(row["n"]) in skip_ns for row in selected):
            raise Probe60GenerationError("partial generation selected a skipped N value")
        return selected
    if len(rows) != 60:
        raise Probe60GenerationError("full generation manifest selection count is {}".format(len(rows)))
    return rows


def validate_generation_selection(selected_rows: list[dict[str, object]], mapping_config: dict) -> None:
    if ONLY_GENERATE_ONE_PILOT_CASE:
        if mapping_config.get("allow_full60_generation") is not False:
            raise Probe60GenerationError("pilot generation requires allow_full60_generation false")
        if len(selected_rows) != 1:
            raise Probe60GenerationError("pilot generation selected row count is {}".format(len(selected_rows)))
        row = selected_rows[0]
        if int(row["n"]) != PILOT_N or row["strategy_name"] != PILOT_STRATEGY:
            raise Probe60GenerationError("pilot generation selected case is not N12_A01")
    else:
        if mapping_config.get("scope") != "full60_generation":
            raise Probe60GenerationError("full generation requires full60_generation scope")
        if mapping_config.get("allow_full60_generation") is not True:
            raise Probe60GenerationError("full generation requires allow_full60_generation true")
        target_ns = [int(n) for n in TARGET_N_VALUES]
        skip_ns = [int(n) for n in SKIP_N_VALUES]
        if target_ns:
            selected_ns = {int(row["n"]) for row in selected_rows}
            if selected_ns != set(target_ns):
                raise Probe60GenerationError(
                    "partial generation selected N values {}; expected {}".format(
                        sorted(selected_ns), target_ns
                    )
                )
            if any(int(row["n"]) in skip_ns for row in selected_rows):
                raise Probe60GenerationError("partial generation selected skipped N values")
            expected_count = 15 * len(target_ns)
            if len(selected_rows) != expected_count:
                raise Probe60GenerationError(
                    "partial generation selected row count {}; expected {}".format(
                        len(selected_rows), expected_count
                    )
                )
        elif len(selected_rows) != 60:
            raise Probe60GenerationError("full generation selected row count is {}".format(len(selected_rows)))


def validate_base_mesh_audit_for_targets(selected_rows: list[dict[str, object]]) -> None:
    target_ns = sorted({int(row["n"]) for row in selected_rows})
    audit_ns = [n for n in target_ns if n in (16, 24, 40)]
    if not audit_ns:
        return
    if not BASE_MESH_AUDIT_SUMMARY.exists():
        raise Probe60GenerationError(
            "missing corrected-base mesh audit summary: {}".format(BASE_MESH_AUDIT_SUMMARY)
        )
    summary = json.loads(BASE_MESH_AUDIT_SUMMARY.read_text(encoding="utf-8"))
    rows_by_n = {int(row["n"]): row for row in summary.get("rows", [])}
    failed = []
    for n in audit_ns:
        row = rows_by_n.get(n)
        if not row or row.get("verdict") != "PASS_BASE_MESH_READY":
            failed.append(n)
            continue
        if float(row.get("total_node_count", 0)) <= 0 or float(row.get("total_element_count", 0)) <= 0:
            failed.append(n)
    if failed:
        raise Probe60GenerationError(
            "corrected-base mesh audit failed or missing for N values: {}".format(failed)
        )


def write_dry_run_log(rows: list[dict[str, object]], mapping_status: str, evidence: dict, config: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_generate_rows(rows, config)
    path = OUTPUT_DIR / "probe60_generator_dry_run_summary.json"
    summary = {
        "mode": MODE,
        "allow_generic_export_without_reorder": ALLOW_GENERIC_EXPORT_WITHOUT_REORDER,
        "allow_user_confirmed_heat_mapping": ALLOW_USER_CONFIRMED_HEAT_MAPPING,
        "only_generate_one_pilot_case": ONLY_GENERATE_ONE_PILOT_CASE,
        "pilot_n": PILOT_N,
        "pilot_strategy": PILOT_STRATEGY,
        "case_count": len(rows),
        "selected_generate_count_if_generate_mode": len(selected),
        "heat_load_mapping_status": mapping_status,
        "heat_mapping_config": str(HEAT_MAPPING_CONFIG),
        "heat_mapping_config_allow_generate_when_confirmed": bool(
            config.get("allow_generate_when_confirmed")
        ),
        "heat_mapping_evidence": evidence,
        "generate_mode_safe": bool(ALLOW_USER_CONFIRMED_HEAT_MAPPING),
        "final_cooling_step_name": config["final_cooling_step_name"],
        "final_cooling_duration_seconds": config["final_cooling_duration_seconds"],
        "final_cooling_initial_increment": config["final_cooling_initial_increment"],
        "final_cooling_max_increment_size": config["final_cooling_max_increment_size"],
        "note": (
            "Dry run only. No CAE, INP, JNL, ODB, datacheck, solver, enqueue, "
            "or abqjobpilot actions were run."
        ),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def validate_heat_sets_exist_in_model(model, n: int, pattern: str) -> dict[int, object]:  # noqa: ANN001
    root = model.rootAssembly
    found: dict[int, object] = {}
    matching_names: set[str] = set()
    for track in range(n):
        name = pattern.format(track=track)
        if name in root.sets:
            found[track] = root.sets[name]
            matching_names.add(name)
            continue
        for instance in root.instances.values():
            if name in instance.sets:
                found[track] = instance.sets[name]
                matching_names.add(name)
                break
        if track not in found:
            raise Probe60GenerationError("missing heat set {} in model".format(name))

    extra_names = [
        name
        for name in list(root.sets.keys())
        if re.fullmatch(r"set_body_heat_\d+", name)
    ]
    for instance in root.instances.values():
        extra_names.extend(
            name for name in list(instance.sets.keys()) if re.fullmatch(r"set_body_heat_\d+", name)
        )
    if len(set(extra_names)) != n:
        raise Probe60GenerationError(
            "expected exactly {} heat sets in model, found {}".format(n, sorted(set(extra_names)))
        )
    return found


def infer_existing_template_objects(model) -> dict[str, object]:  # noqa: ANN001
    config = load_heat_mapping_config()
    scan_name = config["template_scan_step"]
    cool_name = config["template_cool_step"]
    load_name = config["template_load"]
    missing = []
    if scan_name not in model.steps:
        missing.append(scan_name)
    if cool_name not in model.steps:
        missing.append(cool_name)
    if load_name not in model.loads:
        missing.append(load_name)
    if missing:
        raise Probe60GenerationError("missing template objects in model: {}".format(missing))
    return {
        "scan_step": model.steps[scan_name],
        "cool_step": model.steps[cool_name],
        "template_load": model.loads[load_name],
        "scan_step_type": model.steps[scan_name].__class__.__name__,
        "cool_step_type": model.steps[cool_name].__class__.__name__,
        "load_type": model.loads[load_name].__class__.__name__,
    }


def try_read_template_load_magnitude(template_load) -> dict[str, object]:  # noqa: ANN001
    attempted = ("magnitude",)
    for attr in attempted:
        if hasattr(template_load, attr):
            value = getattr(template_load, attr)
            if value is not None:
                return {
                    "object_magnitude_readable": True,
                    "object_magnitude": value,
                    "attempted_attribute_names": attempted,
                }
    return {
        "object_magnitude_readable": False,
        "object_magnitude": None,
        "attempted_attribute_names": attempted,
        "load_object_type": template_load.__class__.__name__,
        "load_object_repr": repr(template_load),
    }


def resolve_confirmed_mapping_values(model, mapping_config: dict) -> dict[str, object]:  # noqa: ANN001
    required = (
        "body_heat_flux_magnitude",
        "scan_duration_seconds",
        "cool_duration_seconds",
        "final_cooling_duration_seconds",
    )
    missing = [key for key in required if key not in mapping_config]
    if missing:
        raise Probe60GenerationError("confirmed config missing resolver values: {}".format(missing))
    if mapping_config.get("allow_generate_when_confirmed") is not True:
        raise Probe60GenerationError("confirmed config does not allow pilot generation")

    template_name = mapping_config["template_load"]
    if template_name not in model.loads:
        raise Probe60GenerationError("missing template load {}".format(template_name))
    template_load = model.loads[template_name]
    if template_load.__class__.__name__ != mapping_config["load_type"]:
        raise Probe60GenerationError(
            "template load type {} does not match config {}".format(
                template_load.__class__.__name__, mapping_config["load_type"]
            )
        )
    object_info = try_read_template_load_magnitude(template_load)
    return {
        "body_heat_flux_magnitude": float(mapping_config["body_heat_flux_magnitude"]),
        "scan_duration_seconds": float(mapping_config["scan_duration_seconds"]),
        "cool_duration_seconds": float(mapping_config["cool_duration_seconds"]),
        "final_cooling_duration_seconds": float(mapping_config["final_cooling_duration_seconds"]),
        "final_cooling_initial_increment": float(mapping_config["final_cooling_initial_increment"]),
        "final_cooling_max_increment_size": float(mapping_config["final_cooling_max_increment_size"]),
        "config_magnitude_used": float(mapping_config["body_heat_flux_magnitude"]),
        "object_magnitude_readable": object_info["object_magnitude_readable"],
        "object_magnitude": object_info["object_magnitude"],
        "attempted_attribute_names": object_info["attempted_attribute_names"],
        "load_object_type": object_info.get("load_object_type", template_load.__class__.__name__),
        "load_object_repr": object_info.get("load_object_repr", repr(template_load)),
        "final_magnitude_source": "confirmed_config",
        "scan_duration_source": "confirmed_config",
        "cool_duration_source": "confirmed_config",
        "final_cooling_duration_source": "confirmed_config",
        "final_cooling_initial_increment_source": "confirmed_config",
        "final_cooling_max_increment_source": "confirmed_config",
    }


def _step_common_kwargs(template_step) -> dict:  # noqa: ANN001
    keys = ("timePeriod", "initialInc", "minInc", "maxInc", "maxNumInc", "deltmx", "nlgeom")
    return {key: getattr(template_step, key) for key in keys if hasattr(template_step, key)}


def _create_step_from_template(model, template_step, name: str, previous: str, forced_time_period=None, forced_initial_inc=None, forced_max_inc=None) -> None:  # noqa: ANN001,E501
    step_type = template_step.__class__.__name__
    kwargs = _step_common_kwargs(template_step)
    if forced_time_period is not None:
        kwargs["timePeriod"] = forced_time_period
    if forced_initial_inc is not None:
        kwargs["initialInc"] = forced_initial_inc
    if forced_max_inc is not None:
        kwargs["maxInc"] = forced_max_inc
    if step_type == "CoupledTempDisplacementStep":
        model.CoupledTempDisplacementStep(name=name, previous=previous, **kwargs)
    elif step_type == "HeatTransferStep":
        model.HeatTransferStep(name=name, previous=previous, **kwargs)
    elif step_type == "StaticStep":
        model.StaticStep(name=name, previous=previous, **kwargs)
    else:
        raise Probe60GenerationError("unsupported step template type {}".format(step_type))


def _set_existing_step_time_period(model, step_name: str, duration: float) -> None:  # noqa: ANN001
    if step_name not in model.steps:
        raise Probe60GenerationError("missing existing step {}".format(step_name))
    try:
        model.steps[step_name].setValues(timePeriod=duration)
    except Exception as exc:
        raise Probe60GenerationError(
            "failed to set {} timePeriod to {}: {}".format(step_name, duration, exc)
        )


def create_scan_and_cool_steps_from_template(model, n: int, scan_order: list[int], mapping_config: dict, resolved_values: dict[str, object]) -> list[dict[str, object]]:  # noqa: ANN001,E501
    validate_scan_order(n, scan_order)
    template_info = infer_existing_template_objects(model)
    scan_template = template_info["scan_step"]
    cool_template = template_info["cool_step"]
    records: list[dict[str, object]] = []
    previous = "Initial"

    for seq, track in enumerate(scan_order):
        scan_name = mapping_config["scan_step_pattern"].format(seq=seq)
        cool_name = mapping_config["cool_step_pattern"].format(seq=seq)
        if seq == 0:
            if scan_name not in model.steps or cool_name not in model.steps:
                raise Probe60GenerationError("template step names for seq 0 are not present")
            _set_existing_step_time_period(
                model, scan_name, float(resolved_values["scan_duration_seconds"])
            )
            _set_existing_step_time_period(
                model, cool_name, float(resolved_values["cool_duration_seconds"])
            )
        else:
            if scan_name in model.steps or cool_name in model.steps:
                raise Probe60GenerationError("target step already exists: {} or {}".format(scan_name, cool_name))
            _create_step_from_template(
                model,
                scan_template,
                scan_name,
                previous,
                forced_time_period=float(resolved_values["scan_duration_seconds"]),
            )
            _create_step_from_template(
                model,
                cool_template,
                cool_name,
                scan_name,
                forced_time_period=float(resolved_values["cool_duration_seconds"]),
            )
        records.append({"seq": seq, "track": track, "scan_step": scan_name, "cool_step": cool_name})
        previous = cool_name
    return records


def create_or_copy_body_heat_flux_for_sequence(model, seq: int, track: int, mapping_config: dict, heat_regions: dict[int, object], resolved_values: dict[str, object]):  # noqa: ANN001,E501
    template_name = mapping_config["template_load"]
    load_name = mapping_config["load_pattern"].format(seq=seq)
    scan_step = mapping_config["scan_step_pattern"].format(seq=seq)
    target_region = heat_regions[track]
    if template_name not in model.loads:
        raise Probe60GenerationError("missing template load {}".format(template_name))
    template_load = model.loads[template_name]
    if template_load.__class__.__name__ != mapping_config["load_type"]:
        raise Probe60GenerationError(
            "unsupported heat load type {}; expected {}".format(
                template_load.__class__.__name__, mapping_config["load_type"]
            )
        )
    magnitude = float(resolved_values["body_heat_flux_magnitude"])

    if seq == 0:
        if load_name != template_name:
            raise Probe60GenerationError("seq 0 load name must match template load")
        try:
            template_load.setValues(region=target_region, magnitude=magnitude)
        except Exception as exc:
            if track != 0:
                raise Probe60GenerationError(
                    "failed to retarget template load {} to track {}: {}".format(
                        load_name, track, exc
                    )
                )
            template_load.setValues(magnitude=magnitude)
        return template_load

    if load_name in model.loads:
        raise Probe60GenerationError("target load already exists: {}".format(load_name))
    return model.BodyHeatFlux(
        createStepName=scan_step,
        magnitude=magnitude,
        name=load_name,
        region=target_region,
    )


def deactivate_heat_loads_outside_target_steps(model, mapping_config: dict, sequence_records: list[dict[str, object]]) -> list[dict[str, str]]:  # noqa: ANN001,E501
    deactivation_records = []
    for record in sequence_records:
        seq = int(record["seq"])
        load_name = mapping_config["load_pattern"].format(seq=seq)
        cool_step = str(record["cool_step"])
        if load_name not in model.loads:
            raise Probe60GenerationError("cannot deactivate missing load {}".format(load_name))
        try:
            model.loads[load_name].deactivate(cool_step)
        except Exception as exc:  # Abaqus raises if the template load is already inactive.
            if seq != 0:
                raise
            deactivation_records.append(
                {
                    "load": load_name,
                    "deactivated_in": cool_step,
                    "note": "template load was already inactive in {}; {}".format(
                        cool_step, exc
                    ),
                }
            )
            continue
        deactivation_records.append({"load": load_name, "deactivated_in": cool_step})
    return deactivation_records


def append_final_cooling_step(model, mapping_config: dict, sequence_records: list[dict[str, object]], resolved_values: dict[str, object]) -> dict[str, object]:  # noqa: ANN001,E501
    final_name = mapping_config["final_cooling_step_name"]
    if final_name in model.steps:
        raise Probe60GenerationError("final cooling step already exists")
    if not sequence_records:
        raise Probe60GenerationError("no sequence records available for final cooling")
    cool_template = model.steps[mapping_config["template_cool_step"]]
    previous = str(sequence_records[-1]["cool_step"])
    _create_step_from_template(
        model,
        cool_template,
        final_name,
        previous,
        forced_time_period=float(resolved_values["final_cooling_duration_seconds"]),
        forced_initial_inc=float(resolved_values["final_cooling_initial_increment"]),
        forced_max_inc=float(resolved_values["final_cooling_max_increment_size"]),
    )
    final_step = model.steps[final_name]
    time_period = getattr(final_step, "timePeriod", None)
    initial_inc = getattr(final_step, "initialInc", None)
    max_inc = getattr(final_step, "maxInc", None)
    if float(time_period) != float(resolved_values["final_cooling_duration_seconds"]):
        raise Probe60GenerationError("final cooling timePeriod verification failed")
    if float(initial_inc) != float(resolved_values["final_cooling_initial_increment"]):
        raise Probe60GenerationError("final cooling initialInc verification failed")
    if float(max_inc) != float(resolved_values["final_cooling_max_increment_size"]):
        raise Probe60GenerationError("final cooling maxInc verification failed")
    return {
        "final_cooling_step_name": final_name,
        "final_cooling_step_created": True,
        "final_cooling_timePeriod": float(time_period),
        "final_cooling_initialInc": float(initial_inc),
        "final_cooling_maxInc": float(max_inc),
        "final_cooling_previous": previous,
    }


def validate_generated_mapping_before_write(model, n: int, scan_order: list[int], mapping_config: dict, sequence_records: list[dict[str, object]], deactivation_records: list[dict[str, str]], resolved_values: dict[str, object], final_cooling_record: dict[str, object]) -> None:  # noqa: ANN001,E501
    validate_scan_order(n, scan_order)
    if len(sequence_records) != n:
        raise Probe60GenerationError("sequence record count does not equal N")
    if len(deactivation_records) != n:
        raise Probe60GenerationError("deactivation record count does not equal N")
    for record in sequence_records:
        seq = int(record["seq"])
        track = int(record["track"])
        expected_set = mapping_config["track_set_pattern"].format(track=track)
        load_name = mapping_config["load_pattern"].format(seq=seq)
        if record["scan_step"] not in model.steps:
            raise Probe60GenerationError("missing generated scan step {}".format(record["scan_step"]))
        if record["cool_step"] not in model.steps:
            raise Probe60GenerationError("missing generated cool step {}".format(record["cool_step"]))
        if load_name not in model.loads:
            raise Probe60GenerationError("missing generated heat load {}".format(load_name))
        if not expected_set.startswith("set_body_heat_"):
            raise Probe60GenerationError("unexpected heat set name {}".format(expected_set))
    final_name = mapping_config["final_cooling_step_name"]
    if final_name not in model.steps:
        raise Probe60GenerationError("missing {}".format(final_name))
    if getattr(model.steps[final_name], "timePeriod", None) != float(
        resolved_values["final_cooling_duration_seconds"]
    ):
        raise Probe60GenerationError("final cooling duration is not 1200.0 seconds")
    if getattr(model.steps[final_name], "initialInc", None) != float(
        resolved_values["final_cooling_initial_increment"]
    ):
        raise Probe60GenerationError("final cooling initialInc is not 0.01")
    if getattr(model.steps[final_name], "maxInc", None) != float(
        resolved_values["final_cooling_max_increment_size"]
    ):
        raise Probe60GenerationError("final cooling maxInc is not 60.0")
    if not final_cooling_record.get("final_cooling_step_created"):
        raise Probe60GenerationError("final cooling creation record missing")
    if final_cooling_record.get("final_cooling_previous") != sequence_records[-1]["cool_step"]:
        raise Probe60GenerationError("final cooling is not after the last cool step")
    deactivated_loads = {item["load"] for item in deactivation_records}
    expected_loads = {mapping_config["load_pattern"].format(seq=i) for i in range(n)}
    if deactivated_loads != expected_loads:
        raise Probe60GenerationError("not all scan loads were deactivated in cool steps")


def write_case_generation_log(case_row: dict[str, object], status: str, notes: dict | str) -> Path:
    log_path = Path(str(case_row["expected_jnl"])).with_suffix(".generation_log.json")
    payload = {
        "status": status,
        "case": {
            "n": case_row["n"],
            "strategy_name": case_row["strategy_name"],
            "job_name": case_row["job_name"],
            "expected_cae": case_row["expected_cae"],
            "expected_inp": case_row["expected_inp"],
        },
        "notes": notes,
        "solver_submitted": False,
        "datacheck_run": False,
        "abqjobpilot_run": False,
    }
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return log_path


def write_case_cae_and_inp(model, case_row: dict[str, object]) -> None:  # noqa: ANN001
    from abaqus import mdb  # type: ignore
    from abaqusConstants import OFF  # type: ignore

    case_dir = Path(str(case_row["case_dir"]))
    expected_cae = Path(str(case_row["expected_cae"]))
    expected_inp = Path(str(case_row["expected_inp"]))
    job_name = str(case_row["job_name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(str(case_dir))
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
    mdb.Job(name=job_name, model=model.name)
    mdb.saveAs(pathName=str(expected_cae))
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    if not expected_inp.exists() or expected_inp.stat().st_size <= 0:
        raise Probe60GenerationError("expected INP was not written: {}".format(expected_inp))


def generate(rows: list[dict[str, object]], mapping_config: dict) -> None:
    if ALLOW_GENERIC_EXPORT_WITHOUT_REORDER:
        raise Probe60GenerationError("generic export without scan-order remapping is forbidden")
    if not ALLOW_USER_CONFIRMED_HEAT_MAPPING:
        raise Probe60GenerationError(
            "generate mode blocked: ALLOW_USER_CONFIRMED_HEAT_MAPPING is False"
        )
    if not ONLY_GENERATE_ONE_PILOT_CASE and not mapping_config.get("allow_generate_when_confirmed"):
        raise Probe60GenerationError(
            "all-60 generation blocked: config allow_generate_when_confirmed is false"
        )

    from abaqus import mdb, openMdb  # type: ignore

    selected_rows = select_generate_rows(rows, mapping_config)
    validate_generation_selection(selected_rows, mapping_config)
    validate_base_mesh_audit_for_targets(selected_rows)
    for row in selected_rows:
        n = int(row["n"])
        validate_case_paths({key: str(row[key]) for key in ("case_dir", "scan_order_json", "expected_cae", "expected_inp", "expected_jnl")}, n)
        scan_order = list(row["scan_order"])
        openMdb(pathName=str(BASES[n]))
        model = mdb.models["Model-1"]
        resolved_values = resolve_confirmed_mapping_values(model, mapping_config)
        if resolved_values["final_magnitude_source"] != "confirmed_config":
            raise Probe60GenerationError("BodyHeatFlux magnitude source is not confirmed_config")
        if resolved_values["scan_duration_source"] != "confirmed_config":
            raise Probe60GenerationError("scan duration source is not confirmed_config")
        if resolved_values["cool_duration_source"] != "confirmed_config":
            raise Probe60GenerationError("cool duration source is not confirmed_config")
        if resolved_values["final_cooling_duration_source"] != "confirmed_config":
            raise Probe60GenerationError("final cooling duration source is not confirmed_config")
        if resolved_values["final_cooling_initial_increment_source"] != "confirmed_config":
            raise Probe60GenerationError("final cooling initialInc source is not confirmed_config")
        if resolved_values["final_cooling_max_increment_source"] != "confirmed_config":
            raise Probe60GenerationError("final cooling maxInc source is not confirmed_config")
        heat_regions = validate_heat_sets_exist_in_model(
            model, n, mapping_config["track_set_pattern"]
        )
        sequence_records = create_scan_and_cool_steps_from_template(
            model, n, scan_order, mapping_config, resolved_values
        )
        for record in sequence_records:
            create_or_copy_body_heat_flux_for_sequence(
                model,
                int(record["seq"]),
                int(record["track"]),
                mapping_config,
                heat_regions,
                resolved_values,
            )
        deactivation_records = deactivate_heat_loads_outside_target_steps(
            model, mapping_config, sequence_records
        )
        final_cooling_record = append_final_cooling_step(
            model, mapping_config, sequence_records, resolved_values
        )
        validate_generated_mapping_before_write(
            model,
            n,
            scan_order,
            mapping_config,
            sequence_records,
            deactivation_records,
            resolved_values,
            final_cooling_record,
        )
        write_case_cae_and_inp(model, row)
        write_case_generation_log(
            row,
            (
                "GENERATED_PILOT"
                if ONLY_GENERATE_ONE_PILOT_CASE
                else "GENERATED_PARTIAL_N16_N24_N40_CASE"
                if TARGET_N_VALUES
                else "GENERATED_FULL60_CASE"
            ),
            {
                "sequence_records": sequence_records,
                "deactivation_records": deactivation_records,
                "resolved_values": resolved_values,
                "final_cooling": dict(
                    final_cooling_record,
                    final_cooling_heat_loads_inactive=True,
                ),
            },
        )


def main() -> int:
    if MODE not in ALLOWED_MODES:
        raise SystemExit("Invalid MODE {!r}; allowed: {}".format(MODE, ALLOWED_MODES))

    try:
        rows, mapping_status, evidence, config = validate_all_inputs()
        if MODE == "dry_run":
            log_path = write_dry_run_log(rows, mapping_status, evidence, config)
            print("DRY_RUN_COMPLETE")
            print("case_count={}".format(len(rows)))
            print("selected_generate_count_if_generate_mode={}".format(len(select_generate_rows(rows, config))))
            print("heat_load_mapping_status={}".format(mapping_status))
            print("allow_user_confirmed_heat_mapping={}".format(ALLOW_USER_CONFIRMED_HEAT_MAPPING))
            print("dry_run_log={}".format(log_path))
            return 0

        generate(rows, config)
        print("GENERATE_COMPLETE")
        return 0
    except Probe60GenerationError as exc:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fail_path = OUTPUT_DIR / "probe60_generator_failure.json"
        fail_path.write_text(
            json.dumps(
                {
                    "mode": MODE,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "config_path": str(HEAT_MAPPING_CONFIG),
                    "pilot_n": PILOT_N,
                    "pilot_strategy": PILOT_STRATEGY,
                    "only_generate_one_pilot_case": ONLY_GENERATE_ONE_PILOT_CASE,
                    "allow_user_confirmed_heat_mapping": ALLOW_USER_CONFIRMED_HEAT_MAPPING,
                    "allow_generic_export_without_reorder": ALLOW_GENERIC_EXPORT_WITHOUT_REORDER,
                    "target_n_values": TARGET_N_VALUES,
                    "skip_n_values": SKIP_N_VALUES,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("FAIL_PROBE60_GENERATOR")
        print(str(exc))
        print("failure_log={}".format(fail_path))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
