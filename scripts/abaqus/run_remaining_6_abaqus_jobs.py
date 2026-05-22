import subprocess
from pathlib import Path
import datetime
import time
import json
import csv
import shutil
import os


ROOT = Path(r"D:\Projects\RL-LAM-ScanOpt")
SUMMARY_DIR = ROOT / "abaqus-models" / "night_run_remaining_6_summary"
SUMMARY_CSV = SUMMARY_DIR / "run_remaining_6_summary.csv"
SUMMARY_JSON = SUMMARY_DIR / "run_remaining_6_summary.json"
HEARTBEAT_LOG = SUMMARY_DIR / "runner_heartbeat.log"

SUCCESS_PHRASES = [
    "THE ANALYSIS HAS COMPLETED SUCCESSFULLY",
    "Abaqus/Standard completed successfully",
    "JOB COMPLETED SUCCESSFULLY",
    "COMPLETED SUCCESSFULLY",
]

SOLVER_PROCESS_NAMES = [
    "standard.exe",
    "SMASim.exe",
    "pre.exe",
    "package.exe",
]

ARCHIVE_SUFFIXES = [
    ".odb",
    ".sta",
    ".msg",
    ".dat",
    ".log",
    ".com",
    ".prt",
    ".sim",
    ".res",
    ".mdl",
    ".stt",
    ".pac",
    ".sel",
    ".abq",
]

SUMMARY_FIELDS = [
    "strategy_name",
    "job_name",
    "workdir",
    "inp_path",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "return_code",
    "status",
    "success_detected",
    "success_phrase",
    "log_path",
    "sta_path",
    "msg_path",
    "dat_path",
    "odb_path",
    "odb_exists",
    "odb_size_bytes",
    "last_sta_lines",
    "last_msg_lines",
]

JOBS = [
    {
        "strategy_name": "formal_raster_left_to_right",
        "workdir": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_raster_formal_v01"),
        "inp": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_raster_formal_v01\job_lded_stripe_tm_32track_raster_formal_v01.inp"),
        "job_name": "job_lded_stripe_tm_32track_raster_formal_v01",
    },
    {
        "strategy_name": "greedy_maximin_distance",
        "workdir": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_greedy_maximin_v01"),
        "inp": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_greedy_maximin_v01\job_lded_stripe_tm_32track_greedy_maximin_v01.inp"),
        "job_name": "job_lded_stripe_tm_32track_greedy_maximin_v01",
    },
    {
        "strategy_name": "smartscan_proxy_variance",
        "workdir": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_smartscan_proxy_v01"),
        "inp": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_smartscan_proxy_v01\job_lded_stripe_tm_32track_smartscan_proxy_v01.inp"),
        "job_name": "job_lded_stripe_tm_32track_smartscan_proxy_v01",
    },
    {
        "strategy_name": "multi_lag_regular_jump",
        "workdir": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_multilag_jump_v01"),
        "inp": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_multilag_jump_v01\job_lded_stripe_tm_32track_multilag_jump_v01.inp"),
        "job_name": "job_lded_stripe_tm_32track_multilag_jump_v01",
    },
    {
        "strategy_name": "block_interleaved_quarters",
        "workdir": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_block_interleaved_quarters_v01"),
        "inp": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_block_interleaved_quarters_v01\job_lded_stripe_tm_32track_block_interleaved_quarters_v01.inp"),
        "job_name": "job_lded_stripe_tm_32track_block_interleaved_quarters_v01",
    },
    {
        "strategy_name": "center_edge_alternating",
        "workdir": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_center_edge_alternating_v01"),
        "inp": Path(r"D:\Projects\RL-LAM-ScanOpt\abaqus-models\lded_stripe_tm_32track_center_edge_alternating_v01\job_lded_stripe_tm_32track_center_edge_alternating_v01.inp"),
        "job_name": "job_lded_stripe_tm_32track_center_edge_alternating_v01",
    },
]


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def append_heartbeat(message):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with open(HEARTBEAT_LOG, "a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def read_last_lines(path, line_count):
    path = Path(path)
    if not path.exists():
        return ""

    max_bytes = 262144
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        data = handle.read()

    text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-line_count:])


def find_success_phrase(paths):
    phrase_bytes = [(phrase, phrase.upper().encode("utf-8")) for phrase in SUCCESS_PHRASES]

    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        previous_tail = b""
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1048576)
                if not chunk:
                    break
                data = (previous_tail + chunk).upper()
                for phrase, encoded in phrase_bytes:
                    if encoded.upper() in data:
                        return phrase
                previous_tail = data[-256:]

    return ""


def make_summary_row(job, status="pending"):
    workdir = job["workdir"]
    job_name = job["job_name"]
    return {
        "strategy_name": job["strategy_name"],
        "job_name": job_name,
        "workdir": str(workdir),
        "inp_path": str(job["inp"]),
        "start_time": "",
        "end_time": "",
        "elapsed_seconds": "",
        "return_code": "",
        "status": status,
        "success_detected": False,
        "success_phrase": "",
        "log_path": str(workdir / ("run_" + job_name + ".log")),
        "sta_path": str(workdir / (job_name + ".sta")),
        "msg_path": str(workdir / (job_name + ".msg")),
        "dat_path": str(workdir / (job_name + ".dat")),
        "odb_path": str(workdir / (job_name + ".odb")),
        "odb_exists": False,
        "odb_size_bytes": 0,
        "last_sta_lines": "",
        "last_msg_lines": "",
    }


def refresh_artifact_fields(row):
    odb_path = Path(row["odb_path"])
    row["odb_exists"] = odb_path.exists()
    row["odb_size_bytes"] = odb_path.stat().st_size if odb_path.exists() else 0
    row["last_sta_lines"] = read_last_lines(row["sta_path"], 50)
    row["last_msg_lines"] = read_last_lines(row["msg_path"], 80)
    return row


def write_summary(rows):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def parse_bool(text):
    lowered = str(text).strip().lower()
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    raise ValueError("Expected true or false, got: " + str(text))


def parse_args(argv):
    args = {
        "cpus": 12,
        "continue_on_failure": False,
        "dry_run": False,
        "start_from": "",
    }

    index = 1
    while index < len(argv):
        item = argv[index]
        if item == "--dry-run":
            args["dry_run"] = True
            index += 1
        elif item == "--cpus":
            if index + 1 >= len(argv):
                raise ValueError("--cpus requires a value")
            args["cpus"] = int(argv[index + 1])
            index += 2
        elif item == "--continue-on-failure":
            if index + 1 >= len(argv):
                raise ValueError("--continue-on-failure requires true or false")
            args["continue_on_failure"] = parse_bool(argv[index + 1])
            index += 2
        elif item == "--start-from":
            if index + 1 >= len(argv):
                raise ValueError("--start-from requires a job_name")
            args["start_from"] = argv[index + 1]
            index += 2
        else:
            raise ValueError("Unknown argument: " + item)

    return args


def solver_processes_running():
    completed = subprocess.run(
        ["tasklist"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        return True, "tasklist failed: " + completed.stderr.strip()

    output = completed.stdout.lower()
    found = []
    for process_name in SOLVER_PROCESS_NAMES:
        if process_name.lower() in output:
            found.append(process_name)
    return len(found) > 0, ", ".join(found)


def check_required_paths(job):
    if not job["workdir"].exists():
        raise RuntimeError("Workdir does not exist: " + str(job["workdir"]))
    if not job["inp"].exists():
        raise RuntimeError("Input file does not exist: " + str(job["inp"]))


def remove_stale_locks_or_stop(workdir):
    locks = list(Path(workdir).glob("*.lck"))
    if not locks:
        return []

    running, detail = solver_processes_running()
    if running:
        raise RuntimeError("Found .lck and active/unknown Abaqus solver process state: " + detail)

    removed = []
    for lock in locks:
        lock.unlink()
        removed.append(str(lock))
    return removed


def completed_result(job):
    job_name = job["job_name"]
    workdir = job["workdir"]
    odb = workdir / (job_name + ".odb")
    sta = workdir / (job_name + ".sta")
    msg = workdir / (job_name + ".msg")
    success_phrase = find_success_phrase([msg, sta])
    return odb.exists() and odb.stat().st_size > 0 and bool(success_phrase), success_phrase


def archive_old_outputs(job):
    workdir = job["workdir"]
    job_name = job["job_name"]
    candidates = []

    for suffix in ARCHIVE_SUFFIXES:
        candidate = workdir / (job_name + suffix)
        if candidate.exists():
            candidates.append(candidate)

    run_log = workdir / ("run_" + job_name + ".log")
    if run_log.exists():
        candidates.append(run_log)

    if not candidates:
        return ""

    archive_dir = workdir / "_archive" / (job_name + "_" + timestamp())
    archive_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        shutil.move(str(candidate), str(archive_dir / candidate.name))
    return str(archive_dir)


def record_failure_tail(row, reason):
    append_heartbeat("[" + now_iso() + "] FAILED: " + row["job_name"] + " - " + reason)
    append_heartbeat("----- last 50 .sta lines -----")
    append_heartbeat(read_last_lines(row["sta_path"], 50) or "(no .sta lines)")
    append_heartbeat("----- last 80 .msg lines -----")
    append_heartbeat(read_last_lines(row["msg_path"], 80) or "(no .msg lines)")


def write_running_heartbeat(row, start_epoch):
    sta_tail = read_last_lines(row["sta_path"], 5) or "(no .sta lines yet)"
    elapsed = int(time.time() - start_epoch)
    message = (
        "[" + now_iso() + "] current_job=" + row["job_name"]
        + " elapsed_seconds=" + str(elapsed)
        + "\n----- last 5 .sta lines -----\n"
        + sta_tail
        + "\n-----------------------------"
    )
    append_heartbeat(message)


def run_one_job(job, row, cpus, dry_run):
    check_required_paths(job)
    removed_locks = remove_stale_locks_or_stop(job["workdir"])
    for removed_lock in removed_locks:
        append_heartbeat("[" + now_iso() + "] removed stale lock: " + removed_lock)

    already_done, success_phrase = completed_result(job)
    if already_done:
        row["status"] = "skipped_completed"
        row["success_detected"] = True
        row["success_phrase"] = success_phrase
        row["end_time"] = now_iso()
        refresh_artifact_fields(row)
        append_heartbeat("[" + now_iso() + "] skipped completed job: " + job["job_name"])
        return True

    if dry_run:
        row["status"] = "pending"
        refresh_artifact_fields(row)
        append_heartbeat("[" + now_iso() + "] dry-run verified job: " + job["job_name"])
        return True

    archive_dir = archive_old_outputs(job)
    if archive_dir:
        append_heartbeat("[" + now_iso() + "] archived old non-success outputs for " + job["job_name"] + ": " + archive_dir)

    row["status"] = "running"
    row["start_time"] = now_iso()
    start_epoch = time.time()
    refresh_artifact_fields(row)

    abaqus_command = shutil.which("abaqus") or shutil.which("abaqus.bat") or "abaqus"
    command = [
        abaqus_command,
        "job=" + job["job_name"],
        "input=" + job["inp"].name,
        "cpus=" + str(cpus),
        "interactive",
    ]

    display_command = [
        "abaqus",
        "job=" + job["job_name"],
        "input=" + job["inp"].name,
        "cpus=" + str(cpus),
        "interactive",
    ]
    append_heartbeat("[" + now_iso() + "] resolved abaqus command: " + str(abaqus_command))
    append_heartbeat("[" + now_iso() + "] starting: " + " ".join(display_command) + " cwd=" + str(job["workdir"]))
    with open(row["log_path"], "w", encoding="utf-8", errors="replace") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(job["workdir"]),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

        next_heartbeat = time.time() + 300
        while process.poll() is None:
            time.sleep(10)
            if time.time() >= next_heartbeat:
                write_running_heartbeat(row, start_epoch)
                next_heartbeat = time.time() + 300

        return_code = process.returncode

    row["end_time"] = now_iso()
    row["elapsed_seconds"] = int(time.time() - start_epoch)
    row["return_code"] = return_code
    success_phrase = find_success_phrase([row["msg_path"], row["sta_path"]])
    refresh_artifact_fields(row)

    success = (
        return_code == 0
        and bool(row["odb_exists"])
        and int(row["odb_size_bytes"]) > 0
        and bool(success_phrase)
    )
    row["success_detected"] = success
    row["success_phrase"] = success_phrase
    row["status"] = "success" if success else "failed"

    if success:
        append_heartbeat("[" + now_iso() + "] success: " + job["job_name"])
    else:
        reason = (
            "return_code=" + str(return_code)
            + ", odb_exists=" + str(row["odb_exists"])
            + ", odb_size_bytes=" + str(row["odb_size_bytes"])
            + ", success_phrase=" + str(success_phrase)
        )
        record_failure_tail(row, reason)

    return success


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    append_heartbeat("[" + now_iso() + "] runner started")

    try:
        args = parse_args(os.sys.argv)
    except Exception as exc:
        append_heartbeat("[" + now_iso() + "] argument error: " + str(exc))
        print(str(exc))
        return 2

    rows = [make_summary_row(job) for job in JOBS]
    write_summary(rows)

    if args["start_from"]:
        names = [job["job_name"] for job in JOBS]
        if args["start_from"] not in names:
            append_heartbeat("[" + now_iso() + "] invalid --start-from: " + args["start_from"])
            return 2
        start_index = names.index(args["start_from"])
    else:
        start_index = 0

    all_ok = True
    for index, job in enumerate(JOBS):
        if index < start_index:
            continue

        row = rows[index]
        try:
            ok = run_one_job(job, row, args["cpus"], args["dry_run"])
        except Exception as exc:
            row["status"] = "failed"
            row["end_time"] = now_iso()
            row["success_detected"] = False
            refresh_artifact_fields(row)
            record_failure_tail(row, str(exc))
            ok = False

        write_summary(rows)
        if not ok:
            all_ok = False
            if not args["continue_on_failure"]:
                append_heartbeat("[" + now_iso() + "] stopping after failed job: " + job["job_name"])
                break

    write_summary(rows)
    append_heartbeat("[" + now_iso() + "] runner finished all_ok=" + str(all_ok))
    return 0 if all_ok else 1


if __name__ == "__main__":
    os.sys.exit(main())
