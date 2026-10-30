"""Submit and collect self-contained Python probes run by Slurm."""

import base64
import json
import os
import re
import shlex
import subprocess
import time
from contextlib import contextmanager

from .environment import PRIVATE_APP_ENV
from .notes import write_text

JOB_ID = re.compile(r"\d+")
PROBE_ID = re.compile(r"(S\d{3,})/(P\d{3,})")
FILE_SETTLE_SECONDS = 60
ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "REQUEUED",
    "RESIZING",
}


class SubmissionError(ValueError):
    def __init__(self, probe_id, reason):
        self.probe_id = probe_id
        super().__init__(f"Probe {probe_id} submission failed: {reason}")


def _scheduler_env():
    env = os.environ.copy()
    for key in PRIVATE_APP_ENV:
        env.pop(key, None)
    return env


def _command(argv, workspace, env=None):
    try:
        return subprocess.run(
            argv,
            cwd=workspace,
            env=_scheduler_env() if env is None else env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Slurm command could not run: {exc}") from exc


def _save_metadata(path, **fields):
    file = path / "probe.json"
    metadata = json.loads(file.read_text())
    metadata.update(fields)
    write_text(file, json.dumps(metadata, indent=2) + "\n")
    return metadata


def _probe_file(probe, name):
    path = probe / name
    return path.is_file() and path.resolve().is_relative_to(probe.resolve())


@contextmanager
def _locked_workspace(workspace):
    import fcntl

    with (workspace.nocturnomath_path / "slurm.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def submit(
    session,
    code,
    expected,
    time_minutes,
    cpus,
    memory_gb,
    partition,
    gpus,
    account,
    qos,
    result_files,
    *,
    correction_of=None,
):
    """Persist a script before submission and return its probe and Slurm IDs."""
    if not code.strip() or not expected.strip():
        raise ValueError("A Python script and a one-line prediction are required.")
    if any(
        type(value) is not int or value < 1 for value in (time_minutes, cpus, memory_gb)
    ):
        raise ValueError("time_minutes, cpus, and memory_gb must be positive integers.")
    if type(gpus) is not int or gpus < 0:
        raise ValueError("gpus must be a nonnegative integer.")
    for label, value in (("partition", partition), ("account", account), ("qos", qos)):
        if value and not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"{label} must be a Slurm name.")
    reserved = {
        "code.py",
        "job.sh",
        "probe.json",
        "stdout.txt",
        "stderr.txt",
        "output.txt",
    }
    if not isinstance(result_files, list) or any(
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
        or name in reserved
        for name in result_files
    ):
        raise ValueError(
            "result_files must contain simple filenames distinct from probe files."
        )

    workspace = session.workspace
    correction = {}
    if correction_of is not None:
        _, parent = _probe(workspace, correction_of)
        if parent["status"] != "error" or not has_verdict(workspace, correction_of):
            raise ValueError("A correction must follow a failed Slurm probe's verdict.")
        correction["correction_of"] = correction_of
    probe = workspace.start_probe(
        code,
        expected,
        session.model,
        backend="slurm",
        environment=session.environment_record,
        python=str(session.environment.python),
        resources={
            "time_minutes": time_minutes,
            "cpus": cpus,
            "memory_gb": memory_gb,
            "partition": partition,
            "gpus": gpus,
            "account": account,
            "qos": qos,
        },
        result_files=result_files,
        **correction,
    )
    probe_id = f"{workspace.session_id}/{probe.name}"
    session.log_transcript(
        "probe_started",
        probe_id=probe_id,
        expected=expected,
        code=code,
        backend="slurm",
    )
    batch = probe / "job.sh"
    batch.write_text(
        "#!/bin/sh\n"
        + "export NOCTURNOMATH_PROBE_DIR="
        + shlex.quote(str(probe))
        + "\n"
        "exec "
        + shlex.quote(str(session.environment.python))
        + " "
        + shlex.quote(str(probe / "code.py"))
        + "\n"
    )
    argv = [
        "sbatch",
        "--parsable",
        "--export=ALL",
        "--chdir",
        str(workspace.path),
        "--output",
        str(probe / "stdout.txt"),
        "--error",
        str(probe / "stderr.txt"),
        "--time",
        str(time_minutes),
        "--cpus-per-task",
        str(cpus),
        "--mem",
        f"{memory_gb}G",
    ]
    if partition:
        argv.extend(["--partition", partition])
    if gpus:
        argv.extend(["--gres", f"gpu:{gpus}"])
    if account:
        argv.extend(["--account", account])
    if qos:
        argv.extend(["--qos", qos])
    argv.append(str(batch))
    try:
        result = _command(argv, workspace.path, session.environment.process_env())
        if result.returncode:
            raise ValueError(
                result.stderr.strip() or result.stdout.strip() or "sbatch failed"
            )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not JOB_ID.fullmatch(job_id):
            raise ValueError(
                f"sbatch returned no recognizable job ID: {result.stdout.strip()}"
            )
    except ValueError as exc:
        workspace.finish_probe(probe, str(exc), [], "failed", str(exc))
        session.log_transcript("probe_failed", probe_id=probe_id, text=str(exc))
        raise SubmissionError(probe_id, str(exc)) from exc
    _save_metadata(probe, job_id=job_id, status="submitted", submitted=time.time())
    session.log_transcript("slurm_submitted", probe_id=probe_id, job_id=job_id)
    return probe_id, job_id


def _probe(workspace, probe_id):
    match = PROBE_ID.fullmatch(probe_id)
    if not match:
        raise ValueError("probe_id must look like S001/P001.")
    probe = workspace.sessions_path / match[1] / "probes" / match[2]
    if not (probe / "probe.json").is_file():
        raise ValueError(f"Probe {probe_id} does not exist.")
    metadata = json.loads((probe / "probe.json").read_text())
    if metadata.get("backend") != "slurm" or not metadata.get("job_id"):
        raise ValueError(f"Probe {probe_id} is not a submitted Slurm job.")
    return probe, metadata


def check(workspace, probe_id, current_transcript=None):
    """Collect one job under a workspace lock shared by all explorers."""
    with _locked_workspace(workspace):
        metadata, collected = _check_locked(workspace, probe_id)
        original = workspace.transcript_file(probe_id.split("/")[0])
        if metadata["status"] not in ("completed", "error", "failed"):
            previous = next(
                (
                    record
                    for record in reversed(workspace.read_transcript(original))
                    if record.get("kind") == "slurm_status"
                    and record.get("probe_id") == probe_id
                ),
                {},
            )
            if any(
                previous.get(key) != metadata.get(key)
                for key in ("status", "slurm_state")
            ):
                workspace.log_transcript(
                    "slurm_status",
                    transcript_path=original,
                    probe_id=probe_id,
                    job_id=metadata["job_id"],
                    status=metadata["status"],
                    slurm_state=metadata.get("slurm_state"),
                )
        if metadata["status"] in ("completed", "error"):
            transcripts = [original]
            if current_transcript is not None and current_transcript != original:
                transcripts.append(current_transcript)
            probe, _ = _probe(workspace, probe_id)
            fields = {
                "probe_id": probe_id,
                "expected": metadata["expected"],
                "code": (probe / "code.py").read_text(),
                "output": (probe / "output.txt").read_text(),
                "images": [
                    str(path.relative_to(workspace.path))
                    for path in sorted(probe.glob("plot-*.png"))
                ],
                "status": metadata["status"],
                "backend": "slurm",
            }
            for transcript in transcripts:
                if not any(
                    record.get("kind") == "run" and record.get("probe_id") == probe_id
                    for record in workspace.read_transcript(transcript)
                ):
                    workspace.log_transcript(
                        "run", transcript_path=transcript, **fields
                    )
        return metadata, collected


def record_verdict(workspace, probe_id, expected, text, current_transcript):
    """Record one Slurm verdict in its original and checking sessions."""
    with _locked_workspace(workspace):
        _probe(workspace, probe_id)
        original = workspace.transcript_file(probe_id.split("/")[0])
        if any(
            record.get("kind") == "verdict" and record.get("probe_id") == probe_id
            for record in workspace.read_transcript(original)
        ):
            return False
        fields = {"probe_id": probe_id, "expected": expected, "text": text}
        workspace.log_transcript("verdict", transcript_path=original, **fields)
        if current_transcript != original:
            workspace.log_transcript(
                "verdict", transcript_path=current_transcript, **fields
            )
        return True


def has_verdict(workspace, probe_id):
    records = workspace.read_transcript(
        workspace.transcript_file(probe_id.split("/")[0])
    )
    return any(
        record.get("kind") == "verdict" and record.get("probe_id") == probe_id
        for record in records
    )


def needs_review(workspace, probe_id):
    """A verdict alone does not finish an interrupted automatic correction."""
    if not has_verdict(workspace, probe_id):
        return True
    _, metadata = _probe(workspace, probe_id)
    return metadata.get("review_pending", False) and not any(
        job.get("correction_of") == probe_id for job in list_jobs(workspace)
    )


def set_review_pending(workspace, probe_id, pending):
    with _locked_workspace(workspace):
        probe, _ = _probe(workspace, probe_id)
        _save_metadata(probe, review_pending=pending)


@contextmanager
def review_lock(workspace, probe_id):
    """Let only one running app automatically resume a session for review."""
    import fcntl

    path = workspace.transcript_file(probe_id.split("/")[0]).parent
    with (path / "slurm-review.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def result_payload(workspace, probe_id, metadata, image_cap):
    """Read a collected result for either automatic or manual presentation."""
    probe, _ = _probe(workspace, probe_id)
    plots = sorted(probe.glob("plot-*.png"))
    paths = [str(path.relative_to(workspace.path)) for path in plots]
    output = (probe / "output.txt").read_text()
    note = metadata.get("execution_note")
    if note == metadata.get("slurm_state"):
        note = None  # The output already includes the scheduler state.
    return {
        "probe_id": probe_id,
        "backend": "slurm",
        "expected": metadata["expected"],
        "code": (probe / "code.py").read_text(),
        "code_path": str((probe / "code.py").relative_to(workspace.path)),
        "output": f"Probe {probe_id}\nSource: {probe_id}/output.txt\n{output}",
        "text": output,
        "status": "slurm-error" if metadata["status"] == "error" else "completed",
        "execution_note": note,
        "plot_paths": paths,
        "plot_urls": [f"/api/asset?path={path}" for path in paths],
        "plot_images": [
            base64.b64encode(path.read_bytes()).decode() for path in plots[:image_cap]
        ],
    }


def _check_locked(workspace, probe_id):
    """Refresh job state; save output after terminal files become visible."""
    probe, metadata = _probe(workspace, probe_id)
    if metadata["status"] in ("completed", "error", "failed"):
        return metadata, False
    job_id = metadata["job_id"]
    queue = _command(
        ["squeue", "--noheader", "--jobs", job_id, "--format", "%T"], workspace.path
    )
    if queue.returncode == 0 and queue.stdout.strip():
        state = queue.stdout.strip().splitlines()[0].strip().upper()
        if state in ACTIVE_STATES:
            return _save_metadata(
                probe,
                status="queued" if state == "PENDING" else "running",
                slurm_state=state,
                terminal_seen=None,
                checked=time.time(),
            ), False

    account = _command(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            job_id,
            "--format",
            "JobIDRaw,State,ExitCode",
        ],
        workspace.path,
    )
    if account.returncode:
        raise ValueError(
            account.stderr.strip() or "Could not query Slurm job accounting."
        )
    row = next(
        (
            line
            for line in account.stdout.splitlines()
            if line.split("|", 1)[0] == job_id
        ),
        None,
    )
    if row is None:
        return _save_metadata(
            probe,
            status="submitted",
            slurm_state="UNKNOWN",
            terminal_seen=None,
            checked=time.time(),
        ), False
    _, raw_state, exit_code = (row.split("|") + ["", ""])[:3]
    state = (
        raw_state.split()[0].split("+")[0].upper() if raw_state.split() else "UNKNOWN"
    )
    if state == "UNKNOWN":
        return _save_metadata(
            probe,
            status="submitted",
            slurm_state=state,
            terminal_seen=None,
            checked=time.time(),
        ), False
    if state in ACTIVE_STATES:
        return _save_metadata(
            probe,
            status="queued" if state == "PENDING" else "running",
            slurm_state=state,
            terminal_seen=None,
            checked=time.time(),
        ), False

    required = ["stdout.txt", "stderr.txt"]
    if state == "COMPLETED":
        required += metadata.get("result_files", [])
    missing_required = [name for name in required if not _probe_file(probe, name)]
    terminal_seen = metadata.get("terminal_seen") or time.time()
    if missing_required and time.time() - terminal_seen < FILE_SETTLE_SECONDS:
        return _save_metadata(
            probe,
            status="finishing",
            slurm_state=state,
            terminal_seen=terminal_seen,
            checked=time.time(),
        ), False

    stdout = (
        (probe / "stdout.txt").read_text(errors="replace")
        if _probe_file(probe, "stdout.txt")
        else ""
    )
    stderr = (
        (probe / "stderr.txt").read_text(errors="replace")
        if _probe_file(probe, "stderr.txt")
        else ""
    )
    output = f"Slurm state: {state}; exit code: {exit_code}\n\nstdout:\n{stdout}\nstderr:\n{stderr}"
    present = [
        name for name in metadata.get("result_files", []) if _probe_file(probe, name)
    ]
    missing = [name for name in metadata.get("result_files", []) if name not in present]
    if present:
        output += "\nSaved result files:\n" + "\n".join(
            f"{probe_id}/{name}" for name in present
        )
    if missing:
        output += "\nMissing declared result files:\n" + "\n".join(missing)
    missing_output = [
        name for name in missing_required if name in ("stdout.txt", "stderr.txt")
    ]
    if missing_output:
        output += "\nMissing Slurm output files:\n" + "\n".join(missing_output)
    status = (
        "completed"
        if state == "COMPLETED" and exit_code.startswith("0:") and not missing_required
        else "error"
    )
    problems = []
    if state != "COMPLETED":
        problems.append(f"Slurm job ended in {state} (exit code {exit_code}).")
    elif not exit_code.startswith("0:"):
        problems.append(f"Slurm job exited with code {exit_code}.")
    if missing:
        problems.append("Missing declared result files: " + ", ".join(missing) + ".")
    if missing_output:
        problems.append(
            "Missing Slurm output files: " + ", ".join(missing_output) + "."
        )
    workspace.finish_probe(probe, output, [], status, " ".join(problems) or None)
    metadata = _save_metadata(
        probe,
        slurm_state=state,
        exit_code=exit_code,
        terminal_seen=terminal_seen,
        checked=time.time(),
    )
    return metadata, True


def list_jobs(workspace):
    """Return saved Slurm jobs, including jobs from earlier sessions."""
    jobs = []
    for file in sorted(workspace.sessions_path.glob("S*/probes/P*/probe.json")):
        metadata = json.loads(file.read_text())
        if metadata.get("backend") == "slurm" and metadata.get("job_id"):
            jobs.append(
                {
                    "probe_id": f"{file.parent.parent.parent.name}/{file.parent.name}",
                    "job_id": metadata["job_id"],
                    "status": metadata["status"],
                    "slurm_state": metadata.get("slurm_state"),
                    "correction_of": metadata.get("correction_of"),
                }
            )
    return jobs


def pending_jobs(workspace):
    """Reconstruct the monitor's work, including collected but unreviewed jobs."""
    with _locked_workspace(workspace):
        return [
            job
            for job in list_jobs(workspace)
            if needs_review(workspace, job["probe_id"])
        ]
