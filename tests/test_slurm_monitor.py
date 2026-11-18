import asyncio
import json
import re
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from conftest import FakeEnvironment, FakeKernel

from nocturnomath import slurm
from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult
from nocturnomath.runtime import Runtime
from nocturnomath.session import ExplorationSession


def tools(session):
    return {tool.name: tool for tool in session.tools}


class _StagedDiscoveryMixin:
    async def discover_models_staged(self, auth):
        return await self.discover_models(auth), lambda: None


@pytest.fixture(autouse=True)
def research_environment():
    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
    ):
        yield


@pytest.fixture
def cluster(monkeypatch):
    cluster = {"states": {}, "unavailable": False}

    def command(argv, workspace, env=None):
        if argv[0] == "sbatch":
            job_id = str(len(cluster["states"]) + 1)
            cluster["states"][job_id] = "RUNNING"
            return SimpleNamespace(returncode=0, stdout=job_id, stderr="")
        if cluster["unavailable"]:
            raise ValueError("Scheduler unavailable")
        job_id = argv[argv.index("--jobs") + 1]
        state = cluster["states"][job_id]
        if argv[0] == "squeue":
            return SimpleNamespace(
                returncode=0,
                stdout=state if state in slurm.ACTIVE_STATES else "",
                stderr="",
            )
        exit_code = "0:0" if state == "COMPLETED" else "1:0"
        return SimpleNamespace(
            returncode=0, stdout=f"{job_id}|{state}|{exit_code}\n", stderr=""
        )

    monkeypatch.setattr(slurm, "_command", command)
    return cluster


@pytest_asyncio.fixture
async def runtime(session, monkeypatch):
    runtime = Runtime(session)
    runtime._started = True
    # Most tests advance one monitor pass explicitly, without clock-based races.
    monkeypatch.setattr(runtime, "_wake_slurm_monitor", lambda: None)
    yield runtime
    await runtime.drain()
    runtime.shutdown()


@pytest.fixture
def reviewer(monkeypatch):
    calls = []

    async def review(session, probe_id):
        result = await tools(session)["check_slurm_job"].handler({"probe_id": probe_id})
        assert not result.get("is_error")
        calls.append((session, probe_id, result))
        verdict = await tools(session)["verdict"].handler(
            {"text": f"Reviewed the saved output of {probe_id} against its prediction."}
        )
        assert not verdict.get("is_error")
        return True

    monkeypatch.setattr(ExplorationSession, "review_slurm", review)
    return calls


async def submit(session, result_files=()):
    if session.workspace.session_pending:
        session.log_transcript("user", text="Investigate the result.")
    result = await tools(session)["submit_slurm"].handler(
        {
            "code": "print(42)",
            "expected": "prints 42",
            "time_minutes": 10,
            "cpus": 1,
            "memory_gb": 1,
            "result_files": list(result_files),
        }
    )
    assert not result.get("is_error")
    jobs = slurm.list_jobs(session.workspace)
    return jobs[-1]["probe_id"], jobs[-1]["job_id"]


def complete(session, cluster, probe_id, job_id, state="COMPLETED"):
    cluster["states"][job_id] = state
    session_id, number = probe_id.split("/")
    probe = session.workspace.sessions_path / session_id / "probes" / number
    (probe / "stdout.txt").write_text("42\n")
    (probe / "stderr.txt").write_text("" if state == "COMPLETED" else "Stopped early\n")
    return probe


async def finish_reviews(runtime):
    await asyncio.gather(
        *(
            session._current_task
            for session in runtime.agents.values()
            if session._current_task is not None
        )
    )


@pytest.mark.asyncio
async def test_monitor_collects_plots_and_reviews_without_a_user_status_request(
    session, runtime, cluster, reviewer, monkeypatch
):
    events = []
    running = asyncio.Event()
    resolved = asyncio.Event()

    def observe(kind, payload):
        events.append((kind, payload))
        if kind == "probe_update" and payload["status"] == "running":
            running.set()
        if kind == "probe_verdict" and payload["probe_id"] == "S001/P001":
            resolved.set()

    session.subscribe(observe)
    probe_id, job_id = await submit(session, ["plot-1.png"])
    assert [kind for kind, _ in events] == ["probe_start", "probe_update"]
    assert session._pending_verdict is None
    monkeypatch.setattr("nocturnomath.runtime.SLURM_POLL_SECONDS", 0.01)
    monkeypatch.setattr(
        runtime, "_wake_slurm_monitor", Runtime._wake_slurm_monitor.__get__(runtime)
    )
    runtime.start()
    await asyncio.wait_for(running.wait(), 2)
    assert reviewer == []

    # Ordinary kernel probes remain available while the batch probe is unresolved.
    await tools(session)["run"].handler({"code": "print(1)", "expected": "1"})
    await tools(session)["verdict"].handler({"text": "The local result matched."})
    probe = complete(session, cluster, probe_id, job_id)
    (probe / "plot-1.png").write_bytes(b"saved plot")
    await asyncio.wait_for(resolved.wait(), 2)
    await finish_reviews(runtime)

    assert [call[1] for call in reviewer] == [probe_id]
    assert reviewer[0][2]["content"][1]["type"] == "image"
    finished = [payload for kind, payload in events if kind == "probe_finish"]
    assert sum(payload["probe_id"] == probe_id for payload in finished) == 1
    assert finished[-1]["probe_id"] == probe_id
    assert finished[-1]["plot_paths"][0].endswith("S001/probes/P001/plot-1.png")
    assert slurm.pending_jobs(session.workspace) == []
    assert session.kernel.executed_codes == ["print(1)"]


@pytest.mark.asyncio
async def test_results_are_saved_while_busy_and_reviews_wait_for_the_current_turn(
    session, runtime, cluster, reviewer
):
    probe_id, job_id = await submit(session)
    busy = asyncio.Event()
    session._current_task = asyncio.create_task(busy.wait())
    current = session._current_task
    probe = complete(session, cluster, probe_id, job_id)
    await runtime._poll_slurm_jobs()
    assert (probe / "output.txt").exists()
    assert reviewer == []
    assert session._current_task is current

    busy.set()
    await current
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert [call[1] for call in reviewer] == [probe_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["reset", "replace", "close", "workspace"])
async def test_review_revalidates_its_owner_after_publishing_the_result(
    session, runtime, cluster, reviewer, tmp_path, change
):
    probe_id, job_id = await submit(session)
    session._track_session_id("original-sdk-chat")
    complete(session, cluster, probe_id, job_id)
    original_workspace = session.workspace
    original_path = session.workspace_path
    original_agent_id = runtime.primary_agent_id
    if change in ("replace", "close"):
        other_id = runtime.create_agent()
        if change == "replace":
            other = runtime.get_agent(other_id)
            other.log_transcript("user", text="A different question.")
            other_session_id = other.workspace.session_id
            runtime.close_agent(other_id)
    other_path = tmp_path / "other-workspace"
    other_path.mkdir()

    async def change_owner(kind, payload):
        if kind != "probe_finish":
            return
        session.unsubscribe(change_owner)
        if change == "reset":
            runtime.reset_session()
        elif change == "replace":
            runtime.replace_agent(original_agent_id, other_session_id)
        elif change == "close":
            runtime.close_agent(original_agent_id)
        else:
            runtime.open_workspace(other_path)

    session.subscribe(change_owner)
    try:
        await runtime._poll_slurm_jobs()
        assert session._current_task is None
        assert reviewer == []
        assert not slurm.has_verdict(original_workspace, probe_id)
        if change == "workspace":
            runtime.open_workspace(original_path)
        await runtime._poll_slurm_jobs()
        await finish_reviews(runtime)
        assert [call[1] for call in reviewer] == [probe_id]
        owner = reviewer[0][0]
        assert owner.workspace_path == original_path
        assert owner.workspace.session_id == "S001"
        assert owner._sdk_session_id == "original-sdk-chat"
    finally:
        # A broken implementation can leave a review attached to a removed explorer.
        if session not in runtime.agents.values() and session._current_task is not None:
            session._current_task.cancel()
            await asyncio.gather(session._current_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_recovery_waits_instead_of_waking_itself_repeatedly(
    session, runtime, cluster, reviewer, monkeypatch
):
    probe_id, job_id = await submit(session)
    complete(session, cluster, probe_id, job_id)
    runtime.open_workspace(session.workspace_path)
    waiting = asyncio.Event()
    repeated = asyncio.Event()
    resolved = asyncio.Event()
    attempts = []
    loop = asyncio.get_running_loop()

    class ObservedWakeup(asyncio.Event):
        async def wait(self):
            if not self.is_set():
                waiting.set()
            return await super().wait()

    def failing_factory(**kwargs):
        attempts.append(kwargs["workspace_path"])
        if len(attempts) > 1:
            loop.call_soon_threadsafe(repeated.set)
        raise RuntimeError("Kernel unavailable")

    def observe(kind, payload):
        if kind == "probe_verdict":
            resolved.set()

    runtime.subscribe(observe)
    runtime._slurm_wakeup = ObservedWakeup()
    monkeypatch.setattr(runtime, "session_factory", failing_factory)
    monkeypatch.setattr("nocturnomath.runtime.SLURM_POLL_SECONDS", 3600)
    monkeypatch.setattr(
        runtime, "_wake_slurm_monitor", Runtime._wake_slurm_monitor.__get__(runtime)
    )
    waiting_task = asyncio.create_task(waiting.wait())
    repeated_task = asyncio.create_task(repeated.wait())
    runtime.start()
    try:
        done, _ = await asyncio.wait(
            [waiting_task, repeated_task],
            timeout=2,
            return_when=asyncio.FIRST_COMPLETED,
        )
        assert waiting_task in done
        assert not repeated.is_set()
        assert len(attempts) == 1
        assert not slurm.has_verdict(session.workspace, probe_id)

        # A new external wakeup can still retry after the environment is repaired.
        monkeypatch.setattr(runtime, "session_factory", ExplorationSession)
        runtime._wake_slurm_monitor()
        await asyncio.wait_for(resolved.wait(), 2)
        await finish_reviews(runtime)
        assert [call[1] for call in reviewer] == [probe_id]
    finally:
        for task in (waiting_task, repeated_task):
            task.cancel()
        await asyncio.gather(waiting_task, repeated_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_multiple_batch_results_do_not_overwrite_a_kernel_verdict(
    session, runtime, cluster, reviewer
):
    first = await submit(session)
    second = await submit(session)
    await tools(session)["run"].handler({"code": "print(1)", "expected": "1"})
    for probe_id, job_id in (first, second):
        complete(session, cluster, probe_id, job_id)
    await runtime._poll_slurm_jobs()
    assert reviewer == []
    assert session._pending_verdict_probe_id == "S001/P003"
    await tools(session)["verdict"].handler({"text": "Local output matched."})

    for _ in range(2):
        await runtime._poll_slurm_jobs()
        await finish_reviews(runtime)
    assert [call[1] for call in reviewer] == [first[0], second[0]]
    verdicts = [
        record["probe_id"]
        for record in session.workspace.current_records()
        if record["kind"] == "verdict"
    ]
    assert verdicts == ["S001/P003", first[0], second[0]]


@pytest.mark.asyncio
async def test_reopening_recovers_collected_results_in_the_original_chat(
    session, runtime, cluster, reviewer
):
    probe_id, job_id = await submit(session)
    session.log_transcript("meta", sdk_session_id="original-sdk-chat")
    complete(session, cluster, probe_id, job_id)
    slurm.check(session.workspace, probe_id)
    runtime.open_workspace(session.workspace_path)
    assert session.workspace.session_pending

    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    owner, reviewed_id, _ = reviewer[0]
    assert reviewed_id == probe_id
    assert owner is not session
    assert owner.workspace.session_id == "S001"
    assert owner._sdk_session_id == "original-sdk-chat"
    assert owner.kernel.executed_codes == []
    assert session.workspace.session_pending  # The unused new chat stays empty.
    assert len(list(session.workspace.sessions_path.glob("S*"))) == 1
    assert [
        record["text"]
        for record in owner.workspace.current_records()
        if record["kind"] == "user"
    ] == ["Investigate the result."]


@pytest.mark.asyncio
async def test_batch_review_after_reopening_preserves_an_outstanding_kernel_verdict(
    session, runtime, cluster, reviewer
):
    probe_id, job_id = await submit(session)
    await tools(session)["run"].handler({"code": "print(1)", "expected": "prints 1"})
    complete(session, cluster, probe_id, job_id)
    await runtime._poll_slurm_jobs()
    assert session._pending_verdict_probe_id == "S001/P002"
    assert reviewer == []

    runtime.open_workspace(session.workspace_path)
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    owner = reviewer[0][0]
    assert reviewer[0][1] == probe_id
    assert owner._pending_verdict_probe_id == "S001/P002"
    assert owner._pending_verdict == "prints 1"
    blocked = await tools(owner)["run"].handler(
        {"code": "print(2)", "expected": "prints 2"}
    )
    assert blocked["is_error"]
    assert owner.kernel.executed_codes == []

    # The kernel verdict also survives another chat resume after the batch review.
    owner.resume_session("S001")
    assert owner._pending_verdict_probe_id == "S001/P002"
    verdict = await tools(owner)["verdict"].handler(
        {"text": "The local result matched."}
    )
    assert not verdict.get("is_error")
    assert owner._pending_verdict is None
    result = await tools(owner)["run"].handler(
        {"code": "print(2)", "expected": "prints 2"}
    )
    assert not result.get("is_error")
    assert owner.kernel.executed_codes == ["print(2)"]
    assert [
        record["probe_id"]
        for record in owner.workspace.current_records()
        if record["kind"] == "verdict"
    ] == [probe_id, "S001/P002"]


@pytest.mark.asyncio
async def test_reopening_monitors_jobs_that_are_still_queued(
    session, runtime, cluster, reviewer
):
    probe_id, job_id = await submit(session)
    cluster["states"][job_id] = "PENDING"
    runtime.open_workspace(session.workspace_path)
    await runtime._poll_slurm_jobs()
    assert reviewer == []
    assert len(runtime.agents) == 1
    assert slurm.list_jobs(session.workspace)[0]["status"] == "queued"
    complete(session, cluster, probe_id, job_id)
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert reviewer[0][0].workspace.session_id == "S001"
    assert reviewer[0][1] == probe_id


@pytest.mark.asyncio
async def test_submission_interruption_does_not_leave_a_saved_job_unmonitored(
    session, runtime, cluster, reviewer, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()
    scheduler = slurm._command

    def delayed_submission(argv, workspace, env=None):
        if argv[0] == "sbatch":
            entered.set()
            release.wait(timeout=3)
        return scheduler(argv, workspace, env)

    monkeypatch.setattr(slurm, "_command", delayed_submission)
    submission = asyncio.create_task(submit(session))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        submission.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await submission
    job = slurm.list_jobs(session.workspace)[0]
    complete(session, cluster, job["probe_id"], job["job_id"])
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert [call[1] for call in reviewer] == [job["probe_id"]]


@pytest.mark.asyncio
async def test_scheduler_outage_keeps_the_probe_unresolved_and_monitor_recovers(
    session, runtime, cluster, reviewer
):
    probe_id, job_id = await submit(session)
    cluster["unavailable"] = True
    await runtime._poll_slurm_jobs()
    probe = session.workspace.probes_path / "P001"
    assert not (probe / "output.txt").exists()
    assert json.loads((probe / "probe.json").read_text())["status"] == "submitted"
    assert reviewer == []

    cluster["unavailable"] = False
    complete(session, cluster, probe_id, job_id)
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert slurm.has_verdict(session.workspace, probe_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "TIMEOUT"])
async def test_failed_jobs_keep_partial_output_and_receive_a_verdict(
    session, runtime, cluster, reviewer, state
):
    probe_id, job_id = await submit(session)
    probe = complete(session, cluster, probe_id, job_id, state)
    (probe / "plot-1.png").write_bytes(b"partial plot")
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert json.loads((probe / "probe.json").read_text())["status"] == "error"
    assert "Stopped early" in reviewer[0][2]["content"][0]["text"]
    assert reviewer[0][2]["content"][1]["type"] == "image"
    assert slurm.has_verdict(session.workspace, probe_id)


@pytest.mark.asyncio
async def test_interrupted_review_stays_unresolved_and_recovers_after_reopening(
    session, runtime, cluster, reviewer, monkeypatch
):
    probe_id, job_id = await submit(session)
    complete(session, cluster, probe_id, job_id)
    entered = asyncio.Event()
    original_review = ExplorationSession.review_slurm

    async def blocked_review(self, probe_id):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ExplorationSession, "review_slurm", blocked_review)
    await runtime._poll_slurm_jobs()
    await asyncio.wait_for(entered.wait(), 2)
    task = session._current_task
    await session.interrupt()
    await task
    await runtime._poll_slurm_jobs()
    assert session._current_task is None
    assert not slurm.has_verdict(session.workspace, probe_id)

    monkeypatch.setattr(ExplorationSession, "review_slurm", original_review)
    runtime.open_workspace(session.workspace_path)
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert [call[1] for call in reviewer] == [probe_id]


@pytest.mark.asyncio
async def test_another_monitor_owns_the_review_until_its_lock_is_released(
    session, runtime, cluster, reviewer
):
    probe_id, job_id = await submit(session)
    complete(session, cluster, probe_id, job_id)
    with slurm.review_lock(session.workspace, probe_id) as claimed:
        assert claimed
        await runtime._poll_slurm_jobs()
        await finish_reviews(runtime)
    assert reviewer == []
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    await runtime._poll_slurm_jobs()
    assert [call[1] for call in reviewer] == [probe_id]


@pytest.mark.asyncio
async def test_declared_plot_must_arrive_before_review(
    session, runtime, cluster, reviewer
):
    probe_id, job_id = await submit(session, ["plot-1.png"])
    probe = complete(session, cluster, probe_id, job_id)
    await runtime._poll_slurm_jobs()
    assert json.loads((probe / "probe.json").read_text())["status"] == "finishing"
    assert not (probe / "output.txt").exists()
    assert reviewer == []
    (probe / "plot-1.png").write_bytes(b"delayed plot")
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert reviewer[0][2]["content"][1]["type"] == "image"


@pytest.mark.asyncio
async def test_review_resumes_the_sdk_and_records_the_verdict_on_the_original_probe(
    session, runtime, cluster
):
    probe_id, job_id = await submit(session, ["plot-1.png"])
    session._track_session_id("original-conversation", "anthropic")
    probe = complete(session, cluster, probe_id, job_id)
    (probe / "plot-1.png").write_bytes(b"saved plot")
    resumes = []

    class ReviewBackend:
        provider = "anthropic"
        uses_compact_hook = True

        async def discover_models(self, auth):
            return []

        async def run_turn(self, prompt, *, resume_session_id=None, **kwargs):
            resumes.append(resume_session_id)
            assert probe_id in prompt
            result = await tools(session)["check_slurm_job"].handler(
                {"probe_id": probe_id}
            )
            assert result["content"][1]["type"] == "image"
            assert f"{probe_id}/output.txt" in result["content"][0]["text"]
            await tools(session)["verdict"].handler(
                {"text": "It printed 42 as predicted."}
            )
            yield TextBlockDone("Reviewed.")
            yield TurnResult()

        async def compact(self, **kwargs):
            return None

    session.backends["anthropic"] = ReviewBackend()
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert resumes == ["original-conversation"]
    records = session.workspace.current_records()
    assert sum(record["kind"] == "user" for record in records) == 1
    assert any(record["kind"] == "slurm_review" for record in records)
    assert [
        (record["probe_id"], record["text"])
        for record in records
        if record["kind"] == "verdict"
    ] == [(probe_id, "It printed 42 as predicted.")]
    assert session.kernel.executed_codes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_state", ["COMPLETED", "FAILED"])
async def test_failed_job_correction_is_submitted_monitored_and_reviewed(
    session, runtime, cluster, replacement_state
):
    original_id, job_id = await submit(session)
    session._track_session_id("original-conversation", "anthropic")
    original = complete(session, cluster, original_id, job_id, "OUT_OF_MEMORY")
    error = "slurmstepd: error: Detected an out-of-memory event.\n"
    (original / "stderr.txt").write_text(error)
    queries = []
    resumes = []

    class CorrectionBackend:
        provider = "anthropic"
        uses_compact_hook = True

        async def discover_models(self, auth):
            return []

        async def run_turn(self, prompt, *, resume_session_id=None, **kwargs):
            resumes.append(resume_session_id)
            # The automatic continuation must allow an evidence-based resubmission.
            assert "submit_slurm" in prompt
            assert "Stop after reviewing it; do not launch further probes" not in prompt
            target = re.search(r"Runtime event: Slurm probe (S\d+/P\d+)", prompt)[1]
            queries.append(target)
            handlers = tools(session)
            result = await handlers["check_slurm_job"].handler({"probe_id": target})
            assert not result.get("is_error")
            if target == original_id:
                assert "OUT_OF_MEMORY" in result["content"][0]["text"]
                await handlers["verdict"].handler(
                    {"text": "The job exhausted its 1 GB allocation before completing."}
                )
                yield TextBlockDone(
                    "I will rerun the same script with 2 GB of memory to address the recorded OOM failure."
                )
                submitted = await handlers["submit_slurm"].handler(
                    {
                        "code": "print(42)",
                        "expected": "prints 42",
                        "time_minutes": 10,
                        "cpus": 1,
                        "memory_gb": 2,
                    }
                )
                assert not submitted.get("is_error")
            else:
                if replacement_state == "COMPLETED":
                    assert result["content"][1]["type"] == "image"
                    verdict = "The corrected job printed 42 as predicted."
                else:
                    verdict = "The replacement failed too; no further correction is supported by its output. User input is needed."
                await handlers["verdict"].handler({"text": verdict})
                yield TextBlockDone(verdict)
            yield TurnResult()

        async def compact(self, **kwargs):
            return None

    session.backends["anthropic"] = CorrectionBackend()
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    jobs = slurm.list_jobs(session.workspace)
    assert len(jobs) == 2
    replacement_id = jobs[1]["probe_id"]
    replacement_job = jobs[1]["job_id"]
    replacement = session.workspace.probes_path / "P002"
    metadata = json.loads((replacement / "probe.json").read_text())
    assert metadata["resources"]["memory_gb"] == 2
    assert metadata["expected"] == "prints 42"
    assert metadata["correction_of"] == original_id
    assert (replacement / "code.py").read_text() == (original / "code.py").read_text()
    assert queries == [original_id]
    records = session.workspace.current_records()
    verdict_index = next(
        i for i, record in enumerate(records) if record["kind"] == "verdict"
    )
    replacement_index = next(
        i
        for i, record in enumerate(records)
        if record["kind"] == "probe_started" and record["probe_id"] == replacement_id
    )
    assert verdict_index < replacement_index
    failed_output = (original / "output.txt").read_text()

    # The agent yields while the corrected job runs; polling does not call the SDK.
    await runtime._poll_slurm_jobs()
    assert queries == [original_id]
    complete(session, cluster, replacement_id, replacement_job, replacement_state)
    if replacement_state == "COMPLETED":
        (replacement / "plot-1.png").write_bytes(b"replacement plot")
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    await runtime._poll_slurm_jobs()
    assert queries == [original_id, replacement_id]
    assert len(slurm.list_jobs(session.workspace)) == 2
    assert slurm.pending_jobs(session.workspace) == []
    assert all(resume == "original-conversation" for resume in resumes)
    records = session.workspace.current_records()
    assert [
        record["probe_id"] for record in records if record["kind"] == "verdict"
    ] == [original_id, replacement_id]
    assert sum(record["kind"] == "user" for record in records) == 1
    assert (original / "stderr.txt").read_text() == error
    assert (original / "output.txt").read_text() == failed_output
    assert json.loads((original / "probe.json").read_text())["status"] == "error"
    assert session.kernel.executed_codes == []


@pytest.mark.asyncio
async def test_review_without_a_verdict_reports_it_and_does_not_loop(
    session, runtime, cluster, monkeypatch
):
    probe_id, job_id = await submit(session)
    complete(session, cluster, probe_id, job_id)
    calls = []
    errors = []

    async def incomplete_review(probe_id):
        calls.append(probe_id)

    def observe(kind, payload):
        if kind == "error":
            errors.append(payload["message"])

    session.subscribe(observe)
    monkeypatch.setattr(session, "review_slurm", incomplete_review)
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    await runtime._poll_slurm_jobs()
    assert calls == [probe_id]
    assert not slurm.has_verdict(session.workspace, probe_id)
    assert "remains unresolved" in errors[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "point, failure",
    [
        ("verdict", "exception"),
        ("verdict", "cancelled"),
        ("verdict", "sdk_error"),
        ("submission", "exception"),
        ("manual_submission", "exception"),
    ],
)
async def test_interrupted_correction_resumes_without_repeating_verdict_or_submission(
    session, runtime, cluster, point, failure
):
    original_id, job_id = await submit(session)
    original = complete(session, cluster, original_id, job_id, "OUT_OF_MEMORY")
    (original / "stderr.txt").write_text("Out of memory\n")
    queries = []
    interrupted = False

    class CorrectionBackend:
        provider = "anthropic"
        uses_compact_hook = True

        async def discover_models(self, auth):
            return []

        async def run_turn(self, prompt, **kwargs):
            nonlocal interrupted
            target = re.search(r"Runtime event: Slurm probe (S\d+/P\d+)", prompt)[1]
            queries.append(target)
            handlers = tools(runtime.get_agent(runtime.find_agent_for_session("S001")))
            result = await handlers["check_slurm_job"].handler({"probe_id": target})
            assert not result.get("is_error")
            if target == original_id:
                already_reviewed = slurm.has_verdict(session.workspace, original_id)
                if already_reviewed:
                    assert "Existing verdict" in result["content"][0]["text"]
                else:
                    await handlers["verdict"].handler(
                        {
                            "text": "The job ran out of memory; I will increase its allocation."
                        }
                    )
                if point == "submission" or interrupted:
                    result = await handlers["submit_slurm"].handler(
                        {
                            "code": "print(42)",
                            "expected": "prints 42",
                            "time_minutes": 10,
                            "cpus": 1,
                            "memory_gb": 2,
                        }
                    )
                    assert not result.get("is_error")
                if not interrupted:
                    interrupted = True
                    yield SessionStarted("sdk-chat")
                    if failure == "sdk_error":
                        raise RuntimeError(
                            "SDK failed before completing the correction"
                        )
                    if failure == "cancelled":
                        raise asyncio.CancelledError()
                    raise RuntimeError("Connection lost during correction")
            else:
                await handlers["verdict"].handler(
                    {"text": "The replacement printed 42 as predicted."}
                )
            yield TextBlockDone("Correction handled.")
            yield TurnResult()

        async def compact(self, **kwargs):
            return None

    backend = CorrectionBackend()
    session.backends["anthropic"] = backend
    runtime.backends["anthropic"] = backend
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert slurm.has_verdict(session.workspace, original_id)
    assert json.loads((original / "probe.json").read_text())["review_pending"] is True
    assert slurm.needs_review(session.workspace, original_id) == (point != "submission")
    if point == "manual_submission":
        manual = await tools(session)["submit_slurm"].handler(
            {
                "code": "print(42)",
                "expected": "prints 42",
                "time_minutes": 10,
                "cpus": 1,
                "memory_gb": 2,
                "correction_of": original_id,
            }
        )
        assert not manual.get("is_error")
        assert not slurm.needs_review(session.workspace, original_id)

    path = session.workspace_path
    runtime.open_workspace(path)
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    jobs = slurm.list_jobs(session.workspace)
    assert len(jobs) == 2
    assert jobs[1]["correction_of"] == original_id
    assert queries.count(original_id) == (2 if point == "verdict" else 1)
    assert not slurm.needs_review(session.workspace, original_id)

    complete(session, cluster, jobs[1]["probe_id"], jobs[1]["job_id"])
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert len(slurm.list_jobs(session.workspace)) == 2
    assert slurm.pending_jobs(session.workspace) == []
    records = session.workspace.load_session("S001")
    assert [
        record["probe_id"] for record in records if record["kind"] == "verdict"
    ] == [original_id, jobs[1]["probe_id"]]
    assert (original / "stderr.txt").read_text() == "Out of memory\n"


@pytest.mark.asyncio
async def test_automatic_review_stays_on_owning_provider_and_waits_when_unavailable(
    session, runtime, cluster
):
    from nocturnomath.backends import ModelSpec

    anthropic_calls = []
    openai_calls = []
    waiting_notices = []
    openai_available = True

    def record_notice(event_type, payload):
        if event_type == "system_message" and "waiting for" in payload.get("text", ""):
            waiting_notices.append(payload["text"])

    runtime.subscribe(record_notice)

    class AnthropicStub(_StagedDiscoveryMixin):
        provider = "anthropic"
        uses_compact_hook = True

        async def discover_models(self, auth):
            return [
                ModelSpec(
                    id="claude-sonnet-5",
                    raw_id="claude-sonnet-5",
                    provider="anthropic",
                    label="Claude Sonnet 5",
                    efforts=["medium"],
                    is_default=True,
                )
            ]

        async def run_turn(self, prompt, **kwargs):
            anthropic_calls.append({"prompt": prompt, **kwargs})
            yield TextBlockDone("Should not be called.")
            yield TurnResult()

        async def compact(self, **kwargs):
            return None

    class OpenAIStub(_StagedDiscoveryMixin):
        provider = "openai"
        uses_compact_hook = False

        async def discover_models(self, auth):
            if not openai_available:
                return []
            return [
                ModelSpec(
                    id="gpt-5.4",
                    raw_id="gpt-5.4",
                    provider="openai",
                    label="GPT-5.4",
                    efforts=["high"],
                )
            ]

        async def run_turn(self, prompt, *, model, resume_session_id=None, **kwargs):
            openai_calls.append(
                {
                    "prompt": prompt,
                    "model": model,
                    "resume_session_id": resume_session_id,
                    **kwargs,
                }
            )
            owner = runtime.get_agent(runtime.find_agent_for_session("S001"))
            handlers = tools(owner)
            if len(openai_calls) == 1:
                submitted = await handlers["submit_slurm"].handler(
                    {
                        "code": "print(42)",
                        "expected": "prints 42",
                        "time_minutes": 10,
                        "cpus": 1,
                        "memory_gb": 1,
                    }
                )
                assert not submitted.get("is_error")
                yield SessionStarted("oai-thread-1")
                yield TextBlockDone("Submitted from OpenAI.")
            else:
                result = await handlers["check_slurm_job"].handler(
                    {"probe_id": "S001/P001"}
                )
                assert not result.get("is_error")
                await handlers["verdict"].handler(
                    {"text": "Printed 42 as predicted on OpenAI."}
                )
                yield TextBlockDone("Reviewed on OpenAI.")
            yield TurnResult()

        async def compact(self, **kwargs):
            return None

    class EmptyGoogleStub(_StagedDiscoveryMixin):
        provider = "google"
        uses_compact_hook = False

        async def discover_models(self, auth):
            return []

        async def run_turn(self, prompt, **kwargs):
            if False:
                yield TextBlockDone("")

        async def compact(self, **kwargs):
            return None

    runtime.backends = {
        "anthropic": AnthropicStub(),
        "openai": OpenAIStub(),
        "google": EmptyGoogleStub(),
    }
    session.backends = runtime.backends
    await runtime.discover_models()
    runtime.set_session_defaults("claude-sonnet-5", "medium", provider="anthropic")

    assert (
        runtime.start_query(
            "Investigate on OpenAI.", "gpt-5.4", "high", provider="openai"
        )
        == "started"
    )
    await runtime.wait_for_query()
    job = slurm.list_jobs(session.workspace)[0]
    probe_id, job_id = job["probe_id"], job["job_id"]

    # Reopen the workspace while OpenAI is unavailable and Claude is the default.
    openai_available = False
    runtime.open_workspace(session.workspace_path)
    await runtime.discover_models()
    assert runtime.models == ["claude-sonnet-5"]

    complete(runtime.session, cluster, probe_id, job_id)
    await runtime._poll_slurm_jobs()
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)

    assert anthropic_calls == []
    assert len(openai_calls) == 1
    assert len(runtime.agents) == 1
    assert slurm.needs_review(runtime.session.workspace, probe_id)
    assert len(waiting_notices) == 1
    assert "waiting for OpenAI models to become available" in waiting_notices[0]

    # Restore OpenAI availability; automatic review resumes on OpenAI, not Claude.
    openai_available = True
    await runtime.discover_models()
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)

    assert anthropic_calls == []
    assert len(openai_calls) == 2
    assert openai_calls[1]["model"] == "gpt-5.4"
    assert openai_calls[1]["resume_session_id"] == "oai-thread-1"
    owner = runtime.get_agent(runtime.find_agent_for_session("S001"))
    assert owner.model == "gpt-5.4"
    assert slurm.has_verdict(owner.workspace, probe_id)
    assert not slurm.needs_review(owner.workspace, probe_id)


@pytest.mark.asyncio
async def test_reopened_review_after_partial_provider_switch_stays_on_new_owner(
    session, runtime, cluster
):
    from nocturnomath.backends import ModelSpec

    anthropic_calls = []
    openai_calls = []

    class AnthropicStub(_StagedDiscoveryMixin):
        provider = "anthropic"
        uses_compact_hook = True

        async def discover_models(self, auth):
            return [
                ModelSpec(
                    id="claude-sonnet-5",
                    raw_id="claude-sonnet-5",
                    provider="anthropic",
                    label="Claude Sonnet 5",
                    efforts=["medium"],
                    is_default=True,
                )
            ]

        async def run_turn(self, prompt, **kwargs):
            anthropic_calls.append({"prompt": prompt, **kwargs})
            yield SessionStarted("ant-1")
            yield TextBlockDone("Initial Claude turn.")
            yield TurnResult()

        async def compact(self, **kwargs):
            return None

    class OpenAIStub(_StagedDiscoveryMixin):
        provider = "openai"
        uses_compact_hook = False

        async def discover_models(self, auth):
            return [
                ModelSpec(
                    id="gpt-5.4",
                    raw_id="gpt-5.4",
                    provider="openai",
                    label="GPT-5.4",
                    efforts=["high"],
                )
            ]

        async def run_turn(
            self, prompt, *, model, tools=(), resume_session_id=None, **kwargs
        ):
            openai_calls.append(
                {
                    "prompt": prompt,
                    "model": model,
                    "resume_session_id": resume_session_id,
                    **kwargs,
                }
            )
            by_name = {spec.name: spec for spec in tools}
            if len(openai_calls) == 1:
                submitted = await by_name["submit_slurm"].handler(
                    {
                        "code": "print(42)",
                        "expected": "prints 42",
                        "time_minutes": 10,
                        "cpus": 1,
                        "memory_gb": 1,
                    }
                )
                assert not submitted.get("is_error")
                raise RuntimeError("OpenAI CLI disconnected before reporting thread ID")

            result = await by_name["check_slurm_job"].handler({"probe_id": "S001/P001"})
            assert not result.get("is_error")
            await by_name["verdict"].handler(
                {"text": "Reviewed on OpenAI after disconnect."}
            )
            yield SessionStarted("oai-recovered")
            yield TextBlockDone("Reviewed on OpenAI.")
            yield TurnResult()

        async def compact(self, **kwargs):
            return None

    class EmptyGoogleStub(_StagedDiscoveryMixin):
        provider = "google"
        uses_compact_hook = False

        async def discover_models(self, auth):
            return []

        async def run_turn(self, prompt, **kwargs):
            if False:
                yield TextBlockDone("")

        async def compact(self, **kwargs):
            return None

    runtime.backends = {
        "anthropic": AnthropicStub(),
        "openai": OpenAIStub(),
        "google": EmptyGoogleStub(),
    }
    session.backends = runtime.backends
    await runtime.discover_models()

    assert runtime.start_query("Start on Claude.") == "started"
    await runtime.wait_for_query()

    assert (
        runtime.start_query(
            "Switch to OpenAI and submit.",
            "gpt-5.4",
            "high",
            provider="openai",
        )
        == "started"
    )
    await runtime.wait_for_query()

    job = slurm.list_jobs(session.workspace)[0]
    probe_id, job_id = job["probe_id"], job["job_id"]

    runtime.open_workspace(session.workspace_path)
    await runtime.discover_models()
    complete(runtime.session, cluster, probe_id, job_id)
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)

    assert len(anthropic_calls) == 1
    assert len(openai_calls) == 2
    assert openai_calls[1]["resume_session_id"] is None
    assert openai_calls[1]["model"] == "gpt-5.4"
    assert "Initial Claude turn." in openai_calls[1]["prompt"]
    owner = runtime.get_agent(runtime.find_agent_for_session("S001"))
    assert slurm.has_verdict(owner.workspace, probe_id)


@pytest.mark.asyncio
async def test_reopened_review_uses_latest_model_and_effort_on_same_native_thread(
    session, runtime, cluster
):
    from nocturnomath.backends import ModelSpec

    openai_calls = []

    class EmptyAnthropicStub(_StagedDiscoveryMixin):
        provider = "anthropic"
        uses_compact_hook = True

        async def discover_models(self, auth):
            return []

        async def run_turn(self, prompt, **kwargs):
            if False:
                yield TextBlockDone("")

        async def compact(self, **kwargs):
            return None

    class OpenAIStub(_StagedDiscoveryMixin):
        provider = "openai"
        uses_compact_hook = False

        async def discover_models(self, auth):
            return [
                ModelSpec(
                    id="gpt-5.4",
                    raw_id="gpt-5.4",
                    provider="openai",
                    label="GPT-5.4",
                    efforts=["low", "high"],
                    is_default=True,
                ),
                ModelSpec(
                    id="gpt-5.4-mini",
                    raw_id="gpt-5.4-mini",
                    provider="openai",
                    label="GPT-5.4 Mini",
                    efforts=["low", "high"],
                ),
            ]

        async def run_turn(
            self, prompt, *, model, effort=None, resume_session_id=None, **kwargs
        ):
            openai_calls.append(
                {
                    "prompt": prompt,
                    "model": model,
                    "effort": effort,
                    "resume_session_id": resume_session_id,
                    **kwargs,
                }
            )
            owner = runtime.get_agent(runtime.find_agent_for_session("S001"))
            handlers = tools(owner)
            if len(openai_calls) == 1:
                yield SessionStarted("oai-shared-thread")
                yield TextBlockDone("First OpenAI turn.")
            elif len(openai_calls) == 2:
                submitted = await handlers["submit_slurm"].handler(
                    {
                        "code": "print(42)",
                        "expected": "prints 42",
                        "time_minutes": 10,
                        "cpus": 1,
                        "memory_gb": 1,
                    }
                )
                assert not submitted.get("is_error")
                yield TextBlockDone("Submitted from mini.")
            else:
                result = await handlers["check_slurm_job"].handler(
                    {"probe_id": "S001/P001"}
                )
                assert not result.get("is_error")
                await handlers["verdict"].handler(
                    {"text": "Reviewed on gpt-5.4-mini at low effort."}
                )
                yield TextBlockDone("Reviewed on mini.")
            yield TurnResult()

        async def compact(self, **kwargs):
            return None

    class EmptyGoogleStub(_StagedDiscoveryMixin):
        provider = "google"
        uses_compact_hook = False

        async def discover_models(self, auth):
            return []

        async def run_turn(self, prompt, **kwargs):
            if False:
                yield TextBlockDone("")

        async def compact(self, **kwargs):
            return None

    runtime.backends = {
        "anthropic": EmptyAnthropicStub(),
        "openai": OpenAIStub(),
        "google": EmptyGoogleStub(),
    }
    session.backends = runtime.backends
    await runtime.discover_models()

    assert (
        runtime.start_query(
            "Turn 1 on GPT-5.4 high.", "gpt-5.4", "high", provider="openai"
        )
        == "started"
    )
    await runtime.wait_for_query()

    assert (
        runtime.start_query(
            "Turn 2 on GPT-5.4 Mini low.",
            "gpt-5.4-mini",
            "low",
            provider="openai",
        )
        == "started"
    )
    await runtime.wait_for_query()

    job = slurm.list_jobs(session.workspace)[0]
    probe_id, job_id = job["probe_id"], job["job_id"]

    runtime.open_workspace(session.workspace_path)
    await runtime.discover_models()
    complete(runtime.session, cluster, probe_id, job_id)
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)

    assert len(openai_calls) == 3
    assert openai_calls[2]["model"] == "gpt-5.4-mini"
    assert openai_calls[2]["effort"] == "low"
    assert openai_calls[2]["resume_session_id"] == "oai-shared-thread"
    owner = runtime.get_agent(runtime.find_agent_for_session("S001"))
    assert (owner.model, owner.effort) == ("gpt-5.4-mini", "low")
    assert slurm.has_verdict(owner.workspace, probe_id)


@pytest.mark.asyncio
async def test_slurm_review_defers_steer_prompts_and_preserves_null_correction_of(
    session, runtime, cluster
):
    probe_id, job_id = await submit(session)
    complete(session, cluster, probe_id, job_id, state="FAILED")

    async def review_with_user_steer():
        session._slurm_review_probe_id = probe_id
        try:
            await tools(session)["check_slurm_job"].handler({"probe_id": probe_id})
            await tools(session)["verdict"].handler(
                {"text": "Job failed on missing flag; submitting fixed script."}
            )
            # User submits a steer message mid-review: must queue rather than hijack submit_slurm.
            session.enqueue_prompt("Switch topic after review", mode="steer")
            assert session._steerable_prefix_count() == 0
            assert session.pending_prompts_snapshot()[0]["mode"] == "queue"
            res = await tools(session)["submit_slurm"].handler(
                {
                    "code": "print('fixed')",
                    "expected": "prints 42",
                    "time_minutes": 10,
                    "cpus": 1,
                    "memory_gb": 1,
                    "correction_of": None,
                }
            )
            assert not res.get("is_error")
        finally:
            session._slurm_review_probe_id = None

    await review_with_user_steer()
    jobs = slurm.list_jobs(session.workspace)
    assert len(jobs) == 2
    replacement_meta = json.loads(
        (
            session.workspace.sessions_path / "S001" / "probes" / "P002" / "probe.json"
        ).read_text()
    )
    assert replacement_meta["correction_of"] == probe_id
    assert not slurm.needs_review(session.workspace, probe_id)
    assert len(session._pending_prompts) == 1
