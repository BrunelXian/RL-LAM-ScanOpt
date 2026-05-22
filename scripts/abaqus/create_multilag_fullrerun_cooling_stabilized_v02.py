from pathlib import Path
import re
import shutil


ROOT = Path(r"D:\Projects\RL-LAM-ScanOpt")
SRC_DIR = ROOT / "abaqus-models" / "lded_stripe_tm_32track_multilag_jump_v01"
SRC_JOB = "job_lded_stripe_tm_32track_multilag_jump_v01"
SRC_INP = SRC_DIR / (SRC_JOB + ".inp")
SRC_CAE = SRC_DIR / "model_lded_stripe_tm_32track_multilag_jump_v01.cae"
SRC_ACTIVATION = SRC_DIR / "activation_map_lded_stripe_tm_32track_multilag_jump_v01.csv"

DST_DIR = ROOT / "abaqus-models" / "lded_stripe_tm_32track_multilag_jump_v02_fullrerun_cooling_stabilized"
DST_JOB = "job_lded_stripe_tm_32track_multilag_jump_v02_fullrerun_cooling_stabilized"
DST_INP = DST_DIR / (DST_JOB + ".inp")
DST_CAE = DST_DIR / "model_lded_stripe_tm_32track_multilag_jump_v02_fullrerun_cooling_stabilized.cae"
DST_ACTIVATION = DST_DIR / "activation_map_lded_stripe_tm_32track_multilag_jump_v02_fullrerun_cooling_stabilized.csv"
DST_README = DST_DIR / "v02_fullrerun_cooling_stabilized_notes.txt"


def fail(message):
    raise RuntimeError(message)


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def get_step_block(text, step_name):
    pattern = r"(?ims)^\*Step\b[^\n]*\bname=" + re.escape(step_name) + r"\b.*?^\*End Step\b[^\n]*"
    match = re.search(pattern, text)
    if not match:
        fail("Could not find step block: " + step_name)
    return match.group(0), match.start(), match.end()


def extract_interaction_blocks(text):
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip().lower()
        if stripped in ("*sfilm", "*sradiate"):
            start = index
            index += 1
            while index < len(lines):
                next_line = lines[index].strip()
                if next_line.startswith("*") or next_line.startswith("**"):
                    break
                index += 1
            blocks.append("\n".join(lines[start:index]))
        else:
            index += 1
    return blocks


def extract_output_requests(final_block):
    marker = re.search(r"(?im)^\*\* OUTPUT REQUESTS\b.*", final_block)
    if not marker:
        fail("Could not find output requests in original final cooling step.")
    output = final_block[marker.start():]
    output = re.sub(r"(?im)^\*\* OUTPUT REQUESTS\b.*\n?", "", output)
    output = re.sub(r"(?im)^\*Restart,.*\n?", "", output)
    output = re.sub(r"(?im)^\*End Step\b[^\n]*", "", output)
    return output.strip()


def make_replacement_steps(output_requests, interaction_blocks):
    interactions = "\n".join(interaction_blocks).strip()
    if not interactions:
        fail("Could not find original *Sfilm/*Sradiate cooling interaction blocks.")

    return f"""** ----------------------------------------------------------------
**
** STEP: step_cooling_transition_10s
**
*Step, name=step_cooling_transition_10s, nlgeom=YES, inc=999999
*Coupled Temperature-displacement, creep=none, deltmx=50.
0.001, 10.0, 1e-10, 0.01
**
** LOADS
**
** Clear all propagated scan heat fluxes. This keeps load_flux_scan_31 inactive.
*Dsflux, op=NEW
**
** INTERACTIONS
**
** Keep external film cooling and ambient radiation active.
{interactions}
**
** OUTPUT REQUESTS
**
*Restart, write, frequency=1
{output_requests}
*End Step
** ----------------------------------------------------------------
**
** STEP: step_final_cooling_remaining
**
*Step, name=step_final_cooling_remaining, nlgeom=YES, inc=999999
*Coupled Temperature-displacement, creep=none, deltmx=100.
1.0, 1500.0, 1e-10, 60.0
**
** LOADS
**
** Keep all scan heat fluxes inactive during remaining cooling.
*Dsflux, op=NEW
**
** INTERACTIONS
**
** Keep external film cooling and ambient radiation active.
{interactions}
**
** OUTPUT REQUESTS
**
*Restart, write, frequency=1
{output_requests}
*End Step
"""


def validate_new_text(text):
    if "step_final_cooling," in text or "name=step_final_cooling," in text:
        fail("Original step_final_cooling still appears to be present.")
    if text.count("*Step, name=step_scan_31") != 1:
        fail("Expected exactly one step_scan_31.")
    if text.count("*Step, name=step_cooling_transition_10s") != 1:
        fail("Expected exactly one transition cooling step.")
    if text.count("*Step, name=step_final_cooling_remaining") != 1:
        fail("Expected exactly one final cooling remaining step.")
    if "*Dsflux, op=NEW\n**\n** INTERACTIONS" not in text:
        fail("Could not verify inactive scan heat flux marker in transition/final steps.")
    if "*Restart, write, frequency=0" in text:
        fail("Restart write frequency=0 still appears in the new input.")
    if text.count("*Restart, write, frequency=1") != 34:
        fail("Expected 34 restart write requests: 32 scan + 2 cooling.")


def main():
    if not SRC_INP.exists():
        fail("Missing source inp: " + str(SRC_INP))
    DST_DIR.mkdir(parents=True, exist_ok=True)

    source_text = read_text(SRC_INP)
    final_block, final_start, final_end = get_step_block(source_text, "step_final_cooling")
    replace_start = final_start
    prior_separator = source_text.rfind("** ----------------------------------------------------------------", 0, final_start)
    prior_end_step = source_text.rfind("*End Step", 0, final_start)
    if prior_separator > prior_end_step:
        replace_start = prior_separator
    output_requests = extract_output_requests(final_block)
    interactions = extract_interaction_blocks(source_text)
    replacement_steps = make_replacement_steps(output_requests, interactions)

    new_text = source_text[:replace_start] + replacement_steps + source_text[final_end:]
    new_text = new_text.replace("*Restart, write, frequency=0", "*Restart, write, frequency=1")
    validate_new_text(new_text)
    DST_INP.write_text(new_text, encoding="utf-8", newline="\n")

    if SRC_ACTIVATION.exists():
        shutil.copy2(SRC_ACTIVATION, DST_ACTIVATION)
    if SRC_CAE.exists():
        shutil.copy2(SRC_CAE, DST_CAE)

    notes = f"""v02 full rerun cooling stabilized input

Source directory:
{SRC_DIR}

Source job:
{SRC_JOB}

Target job:
{DST_JOB}

Changes:
- Preserved original model definition and all 32 multi_lag scan steps/order.
- Replaced original step_final_cooling with:
  - step_cooling_transition_10s: coupled temp-displacement, 10 s, initialInc 0.001, minInc 1e-10, maxInc 0.01, deltmx 50, inc 999999.
  - step_final_cooling_remaining: coupled temp-displacement, 1500 s, initialInc 1.0, minInc 1e-10, maxInc 60.0, deltmx 100, inc 999999.
- Replaced every '*Restart, write, frequency=0' with '*Restart, write, frequency=1'.
- Both new cooling steps use '*Dsflux, op=NEW' before interactions, so load_flux_scan_31 and all scan heat fluxes are inactive.
- Copied original *Sfilm and *Sradiate blocks into both cooling steps.
- Reused original final-cooling output variables.
"""
    DST_README.write_text(notes, encoding="utf-8")

    print("Created:")
    print("  " + str(DST_INP))
    if DST_ACTIVATION.exists():
        print("  " + str(DST_ACTIVATION))
    if DST_CAE.exists():
        print("  " + str(DST_CAE))
    print("  " + str(DST_README))


if __name__ == "__main__":
    main()
