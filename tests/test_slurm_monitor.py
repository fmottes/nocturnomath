import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from claude_agent_sdk import AssistantMessage, TextBlock
from conftest import FakeEnvironment, FakeKernel

from nocturnomath import slurm
from nocturnomath.runtime import Runtime
from nocturnomath.session import ExplorationSession


def tools(session):
    return {tool.name: tool for tool in session.tools}


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
    session, runtime, cluster, monkeypatch
):
    probe_id, job_id = await submit(session, ["plot-1.png"])
    session._sdk_session_id = "original-conversation"
    probe = complete(session, cluster, probe_id, job_id)
    (probe / "plot-1.png").write_bytes(b"saved plot")
    options_used = []

    class ReviewClient:
        def __init__(self, options):
            options_used.append(options)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def query(self, prompt):
            assert probe_id in prompt

        async def receive_response(self):
            result = await tools(session)["check_slurm_job"].handler(
                {"probe_id": probe_id}
            )
            assert result["content"][1]["type"] == "image"
            assert f"{probe_id}/output.txt" in result["content"][0]["text"]
            await tools(session)["verdict"].handler(
                {"text": "It printed 42 as predicted."}
            )
            yield AssistantMessage(content=[TextBlock("Reviewed.")], model="test")

    monkeypatch.setattr("nocturnomath.session.ClaudeSDKClient", ReviewClient)
    await runtime._poll_slurm_jobs()
    await finish_reviews(runtime)
    assert options_used[0].resume == "original-conversation"
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
