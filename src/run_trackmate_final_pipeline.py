"""Run or monitor final detection, then automatically run Robust Z processing."""

from argparse import ArgumentParser
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def process_exists(pid):
    if sys.platform != "win32":
        try:
            import os
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information, False, int(pid)
    )
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def run_logged(command, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write("\nCOMMAND\n{}\n\n".format(subprocess.list2cmdline(command)))
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code {}; see {}".format(
                completed.returncode, log_path
            )
        )


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=project_root / "outputs" / "TrackMate_final_r2p5_q150",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--detection-workers", type=int, default=6)
    args = parser.parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)

    python = str(Path(sys.executable).resolve())
    detection_status_path = args.result_root / "detection_status.json"
    robust_status_path = args.result_root / "robust_z_status.json"
    pipeline_log = args.result_root / "pipeline_orchestrator.log"

    detection_status = read_json(detection_status_path)
    if not detection_status or detection_status.get("status") != "complete":
        pid_path = args.result_root / "detection_pid.txt"
        existing_pid = None
        if pid_path.is_file():
            try:
                existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
            except ValueError:
                existing_pid = None
        if existing_pid and process_exists(existing_pid):
            print("Monitoring existing detection PID {}.".format(existing_pid), flush=True)
            while True:
                time.sleep(args.poll_seconds)
                detection_status = read_json(detection_status_path)
                if detection_status and detection_status.get("status") == "complete":
                    break
                if detection_status and detection_status.get("status") == "completed_with_failures":
                    raise RuntimeError("Detection completed with failures.")
                if not process_exists(existing_pid):
                    raise RuntimeError(
                        "Detection PID {} exited before status became complete.".format(
                            existing_pid
                        )
                    )
                completed = detection_status.get("dataset_completed", 0) if detection_status else 0
                print("Detection running: {}/40 datasets.".format(completed), flush=True)
        else:
            run_logged(
                [
                    python,
                    "-u",
                    "src/trackmate_final_detection.py",
                    "--workers",
                    str(args.detection_workers),
                    "--java-memory",
                    "8g",
                ],
                pipeline_log,
                project_root,
            )

    robust_status = read_json(robust_status_path)
    if robust_status and robust_status.get("status") == "complete":
        print("Robust Z output is already complete.", flush=True)
        return
    print("Starting full Robust Z and thresholded XML generation.", flush=True)
    run_logged(
        [
            python,
            "-u",
            "src/trackmate_final_robust_z_parallel.py",
            "--result-root",
            str(args.result_root),
            "--workers",
            "6",
        ],
        pipeline_log,
        project_root,
    )
    robust_status = read_json(robust_status_path)
    if not robust_status or robust_status.get("status") != "complete":
        raise RuntimeError("Robust Z command returned without complete status.")
    print("Final TrackMate pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
