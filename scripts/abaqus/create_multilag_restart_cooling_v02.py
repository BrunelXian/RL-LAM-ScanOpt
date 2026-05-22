from pathlib import Path
import re
import shutil
import sys


ROOT = Path(r"D:\Projects\RL-LAM-ScanOpt")
OLD_DIR = ROOT / "abaqus-models" / "lded_stripe_tm_32track_multilag_jump_v01"
OLD_JOB = "job_lded_stripe_tm_32track_multilag_jump_v01"
OLD_INP = OLD_DIR / (OLD_JOB + ".inp")
OLD_STA = OLD_DIR / (OLD_JOB + ".sta")
OLD_MSG = OLD_DIR / (OLD_JOB + ".msg")

NEW_DIR = ROOT / "abaqus-models" / "lded_stripe_tm_32track_multilag_jump_v02_restart_cooling"
NEW_JOB = "job_lded_stripe_tm_32track_multilag_jump_v02_restart_cooling"
NEW_INP = NEW_DIR / (NEW_JOB + ".inp")
COMMAND_TXT = NEW_DIR / "run_restart_command.txt"

TRANSITION_DELT_MX = 50.0
TRANSITION_TIME = 10.0
TRANSITION_INITIAL_INC = 0.001
TRANSITION_MIN_INC = 1e-10
TRANSITION_MAX_INC = 0.01
TRANSITION_MAX_NUM_INC = 999999

FINAL_DELT_MX = 100.0
FINAL_INITIAL_INC = 1.0
FINAL_MIN_INC = 1e-10
FINAL_MAX_INC = 60.0
FINAL_MAX_NUM_INC = 999999
DEFAULT_FINAL_TIME = 1800.0


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def fail(message):
    print("ERROR: " + message)
    raise SystemExit(1)


def require_file(path):
    if not path.exists():
        fail("Required file does not exist: " + str(path))


def collect_restart_files():
    restart_suffixes = [
        ".res",
        ".stt",
        ".mdl",
        ".prt",
        ".sim",
        ".pac",
        ".sel",
        ".abq",
        ".com",
    ]
    found = {}
    for suffix in restart_suffixes:
        files = sorted(OLD_DIR.glob(OLD_JOB + suffix))
        if files:
            found[suffix] = files
    return found


def parse_steps(inp_text):
    steps = []
    for match in re.finditer(r"(?im)^\*Step\b[^\n]*\bname=([^,\s]+)[^\n]*", inp_text):
        steps.append(
            {
                "name": match.group(1),
                "number": len(steps) + 1,
                "start": match.start(),
                "line": inp_text.count("\n", 0, match.start()) + 1,
            }
        )
    return steps


def extract_step_block(inp_text, step_name):
    pattern = r"(?ims)^\*Step\b[^\n]*\bname=" + re.escape(step_name) + r"\b.*?^\*End Step\b[^\n]*"
    match = re.search(pattern, inp_text)
    if not match:
        fail("Could not find step block in original inp: " + step_name)
    return match.group(0)


def parse_final_time(final_step_block):
    for line in final_step_block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("**"):
            continue
        fields = [part.strip() for part in stripped.split(",")]
        if len(fields) >= 2:
            try:
                return float(fields[1])
            except ValueError:
                return DEFAULT_FINAL_TIME
    return DEFAULT_FINAL_TIME


def extract_output_requests(step_block):
    marker = re.search(r"(?im)^\*\* OUTPUT REQUESTS\b.*", step_block)
    if not marker:
        return ""
    output = step_block[marker.start():]
    output = re.sub(r"(?im)^\*Restart,.*\n?", "", output)
    output = re.sub(r"(?im)^\*Output, field[^\n]*", "*Output, field, time interval=60.", output)
    output = re.sub(r"(?im)^NT,\s*RF,\s*U\s*$", "NT, RF, U", output)
    output = re.sub(r"(?im)^PEEQ,\s*S\s*$", "PEEQ, S", output)
    return output.strip()


def extract_cooling_interactions(inp_text):
    lines = inp_text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.lower() in ("*sfilm", "*sradiate"):
            start = index
            index += 1
            while index < len(lines):
                stripped = lines[index].strip()
                if stripped.startswith("*") or stripped.startswith("**"):
                    break
                index += 1
            blocks.append("\n".join(lines[start:index]))
        else:
            index += 1
    return blocks


def sta_has_completed_step(sta_text, step_number):
    accepted_rows = []
    pattern = re.compile(r"^\s+" + re.escape(str(step_number)) + r"\s+\d+\s+\d+\s+", re.MULTILINE)
    for match in pattern.finditer(sta_text):
        line = match.group(0)
        full_line = sta_text[match.start():sta_text.find("\n", match.start())]
        parts = full_line.split()
        if len(parts) >= 9 and "U" not in parts[2]:
            accepted_rows.append(parts)
    if not accepted_rows:
        return False
    last = accepted_rows[-1]
    try:
        step_time = float(last[7])
    except ValueError:
        return False
    return step_time >= 3.59


def sta_shows_final_failure(sta_text, final_step_number):
    final_rows = []
    pattern = re.compile(r"^\s+" + re.escape(str(final_step_number)) + r"\s+", re.MULTILINE)
    for match in pattern.finditer(sta_text):
        full_line = sta_text[match.start():sta_text.find("\n", match.start())]
        final_rows.append(full_line)
    if not final_rows:
        return False
    has_unaccepted = any("U" in row.split()[2] for row in final_rows if len(row.split()) >= 3)
    not_completed = "THE ANALYSIS HAS NOT BEEN COMPLETED" in sta_text.upper()
    return has_unaccepted and not_completed


def write_restart_inp(scan31_step_number, final_time, output_requests, cooling_interactions):
    interaction_text = "\n".join(cooling_interactions).strip()
    if not interaction_text:
        interaction_text = "** No *Sfilm/*Sradiate blocks were found in the original input."

    output_text = output_requests.strip()
    if not output_text:
        output_text = """** OUTPUT REQUESTS
*Output, field, time interval=60.
*Node Output
NT, RF, U
*Element Output, directions=YES
PEEQ, S
*Output, history
*Energy Output
ALLAE, ALLIE, ALLKE, ALLPD, ALLSE, ALLWK, ETOTAL"""

    content = f"""*HEADING
Restart cooling repair for multi_lag_regular_jump
Generated from {OLD_INP}
Old job: {OLD_JOB}
Restart point: end of step_scan_31, original step number {scan31_step_number}
*RESTART, READ, STEP={scan31_step_number}, END STEP
** ----------------------------------------------------------------
**
** STEP: step_cooling_transition_10s
**
*Step, name=step_cooling_transition_10s, nlgeom=YES, inc={TRANSITION_MAX_NUM_INC}
*Coupled Temperature-displacement, creep=none, deltmx={TRANSITION_DELT_MX:g}.
{TRANSITION_INITIAL_INC:g}, {TRANSITION_TIME:g}, {TRANSITION_MIN_INC:g}, {TRANSITION_MAX_INC:g}
**
** LOADS
**
** Clear all propagated scan heat fluxes, including load_flux_scan_31.
*Dsflux, op=NEW
**
** COOLING INTERACTIONS
**
{interaction_text}
**
{output_text}
*End Step
** ----------------------------------------------------------------
**
** STEP: step_final_cooling_remaining
**
*Step, name=step_final_cooling_remaining, nlgeom=YES, inc={FINAL_MAX_NUM_INC}
*Coupled Temperature-displacement, creep=none, deltmx={FINAL_DELT_MX:g}.
{FINAL_INITIAL_INC:g}, {final_time:g}, {FINAL_MIN_INC:g}, {FINAL_MAX_INC:g}
**
** LOADS
**
** Keep scan heat fluxes inactive during remaining cooling.
*Dsflux, op=NEW
**
** COOLING INTERACTIONS
**
{interaction_text}
**
{output_text}
*End Step
"""
    NEW_INP.write_text(content, encoding="utf-8")


def write_command_file():
    relative_oldjob = r"..\lded_stripe_tm_32track_multilag_jump_v01" + "\\" + OLD_JOB
    absolute_oldjob = str(OLD_DIR / OLD_JOB)
    content = f"""cd "{NEW_DIR}"
abaqus oldjob={relative_oldjob} job={NEW_JOB} cpus=12 interactive

If Abaqus does not accept the relative oldjob path, use:
abaqus oldjob="{absolute_oldjob}" job={NEW_JOB} cpus=12 interactive
"""
    COMMAND_TXT.write_text(content, encoding="utf-8")


def main():
    require_file(OLD_INP)
    require_file(OLD_STA)
    require_file(OLD_MSG)

    restart_files = collect_restart_files()
    print("Restart-related files found:")
    for suffix in sorted(restart_files):
        for path in restart_files[suffix]:
            print("  " + suffix + ": " + str(path))

    if ".res" not in restart_files:
        fail(
            "No .res file found for " + OLD_JOB
            + ". Abaqus cannot restart from step_scan_31 with this directory; "
            + "the original run appears to have used '*Restart, write, frequency=0', "
            + "so this case must be rerun from the beginning with restart writes enabled."
        )

    inp_text = read_text(OLD_INP)
    sta_text = read_text(OLD_STA)
    msg_text = read_text(OLD_MSG)

    steps = parse_steps(inp_text)
    step_by_name = {step["name"]: step for step in steps}
    if "step_scan_31" not in step_by_name:
        fail("Could not find step_scan_31 in original inp.")
    if "step_final_cooling" not in step_by_name:
        fail("Could not find step_final_cooling in original inp.")

    scan31_step_number = step_by_name["step_scan_31"]["number"]
    final_step_number = step_by_name["step_final_cooling"]["number"]

    if not sta_has_completed_step(sta_text, scan31_step_number):
        fail("Could not confirm that step_scan_31 completed in the .sta file.")

    if "TOO MANY ATTEMPTS MADE FOR THIS INCREMENT" not in msg_text:
        fail("Could not confirm the expected TOO MANY ATTEMPTS failure in the .msg file.")

    if not sta_shows_final_failure(sta_text, final_step_number):
        fail("Could not confirm that failure occurred in step_final_cooling in the .sta file.")

    final_block = extract_step_block(inp_text, "step_final_cooling")
    output_requests = extract_output_requests(final_block)
    cooling_interactions = extract_cooling_interactions(inp_text)
    final_time = parse_final_time(final_block)

    NEW_DIR.mkdir(parents=True, exist_ok=True)
    write_restart_inp(scan31_step_number, final_time, output_requests, cooling_interactions)
    write_command_file()

    print("")
    print("Generated restart cooling repair input:")
    print("  " + str(NEW_INP))
    print("Generated command file:")
    print("  " + str(COMMAND_TXT))
    print("step_scan_31 step number: " + str(scan31_step_number))
    print("transition cooling: time=10.0, initialInc=0.001, minInc=1e-10, maxInc=0.01, deltmx=50, inc=999999")
    print("final cooling remaining: time=" + str(final_time) + ", initialInc=1.0, minInc=1e-10, maxInc=60.0, deltmx=100, inc=999999")
    print("Copied cooling interaction blocks: " + str(len(cooling_interactions)))


if __name__ == "__main__":
    main()
