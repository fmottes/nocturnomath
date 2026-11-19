"""Transport-agnostic runtime shared by the web app and the terminal client."""

import asyncio
import logging
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any

from . import slurm
from .auth import PROVIDER_TITLES, PROVIDERS, AuthStore, ProviderId
from .backends import DEFAULT_EFFORT, AgentBackend, default_backends
from .session import DEFAULT_PROMPT_MODE, ExplorationSession

logger = logging.getLogger("nocturnomath.runtime")

EVIDENCE_PATH = ".nocturnomath/kb/evidence.md"
THOUGHTS_PATH = ".nocturnomath/kb/thoughts.md"
SLURM_POLL_SECONDS = 30


class Runtime:
    """Own the optional workspace and publish its events to any interface."""

    def __init__(
        self,
        session: ExplorationSession | None = None,
        *,
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int = 600,
        image_cap: int = 2,
        navigator_root: Path | str = ".",
        session_factory: Callable[..., ExplorationSession] = ExplorationSession,
        auth: AuthStore | None = None,
        backends: Mapping[ProviderId, AgentBackend] | None = None,
    ):
        # ``session`` remains the terminal client's single-explorer facade.
        # The web client uses the ordered registry below and always supplies an id.
        self.session = session
        self.agents: dict[str, ExplorationSession] = {}
        self.primary_agent_id: str | None = None
        if backends is not None:
            self.backends: dict[ProviderId, AgentBackend] = dict(backends)
        elif session is not None:
            self.backends = session.backends
        else:
            self.backends = default_backends()
        self._auth = auth or (session.auth if session else AuthStore())
        if session:
            session.auth = self._auth
            session.backends = self.backends
        self.startup_default_model = model
        self.startup_default_effort = effort
        self.model = session.model if session else None
        self.effort = session.effort if session else (effort or DEFAULT_EFFORT)
        self.models: list[str] = []
        self.model_labels: dict[str, str] = {}
        self.model_providers: dict[str, ProviderId] = {}
        self.model_raw_ids: dict[str, str] = {}
        self.model_efforts: dict[str, list[str]] = {}
        self._provider_models: dict[ProviderId, list[str]] = {p: [] for p in PROVIDERS}
        self._provider_default_models: dict[ProviderId, str] = {}
        self._models_discovered = False
        self._auth_revision = 0
        self._discovery_seq = 0
        self._published_discovery_seq = 0
        self.sdk_default_model: str | None = None
        self.timeout_s = session.timeout_s if session else timeout_s
        self.image_cap = session.image_cap if session else image_cap
        self.navigator_root = Path(navigator_root).expanduser().resolve()
        self.session_factory = session_factory
        self.subscribers: list[Callable[[str, dict[str, Any]], Any]] = []
        self._started = False
        self.changing = False
        self._changing_agents: set[str] = set()
        self._creating_agents = 0
        self._discovering_models = 0
        self._agents_lock = threading.RLock()
        self._resuming_sessions: set[str] = set()
        self._agent_subscribers: dict[str, Callable[[str, dict[str, Any]], Any]] = {}
        self._workspace_defaults_pending = session is not None
        self._slurm_task: asyncio.Task | None = None
        self._slurm_wakeup = asyncio.Event()
        self._slurm_seen: dict[tuple[Path, str], tuple] = {}
        self._slurm_review_attempts: set[tuple[Path, str]] = set()
        self._slurm_waiting_notices: set[tuple[Path, str, ProviderId]] = set()
        self._closed_session_meta_cache: dict[
            Path, tuple[int, int, tuple[str | None, str | None, str | None]]
        ] = {}
        if session is not None:
            self.primary_agent_id = self._new_agent_id()
            self.agents[self.primary_agent_id] = session

    @property
    def has_workspace(self) -> bool:
        return bool(self.agents)

    @property
    def auth(self) -> AuthStore:
        """Credential store shared across all open explorers."""
        return self.session.auth if self.session else self._auth

    @auth.setter
    def auth(self, auth: AuthStore):
        self._auth = auth
        self._auth_revision += 1
        for session in self.agents.values():
            session.auth = auth

    def start(self):
        self._started = True
        for agent_id, session in self.agents.items():
            self._subscribe_to_agent(agent_id, session)
        self._wake_slurm_monitor()

    def _wake_slurm_monitor(self):
        if not self._started:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Workspace changes also run in worker threads.
            if self._slurm_task is not None:
                self._slurm_task.get_loop().call_soon_threadsafe(self._slurm_wakeup.set)
            return
        self._slurm_wakeup.set()
        if self._slurm_task is None or self._slurm_task.done():
            self._slurm_task = loop.create_task(self._monitor_slurm())

    async def _monitor_slurm(self):
        while self._started:
            self._slurm_wakeup.clear()
            try:
                await self._poll_slurm_jobs()
            except Exception:
                logger.exception("Could not refresh saved Slurm jobs")
                # Failed recovery may have woken us through transition cleanup.
                # Wait for the next interval or a new external event before retrying.
                self._slurm_wakeup.clear()
            try:
                await asyncio.wait_for(
                    self._slurm_wakeup.wait(), timeout=SLURM_POLL_SECONDS
                )
            except TimeoutError:
                pass

    def _monitoring_workspace(self, workspace):
        return (
            self._started
            and not self.changing
            and self.session is not None
            and self.session.workspace is workspace
        )

    def _slurm_owner_is_current(self, workspace, agent_id, session, probe_id):
        return (
            self._monitoring_workspace(workspace)
            and session is not None
            and self.agents.get(agent_id) is session
            and session.workspace.path == workspace.path
            and not session.workspace.session_pending
            and session.workspace.session_id == probe_id.split("/")[0]
            and agent_id not in self._changing_agents
        )

    def _select_provider_model(
        self,
        provider: ProviderId,
        preferred_model: str | None = None,
        preferred_effort: str | None = None,
    ) -> tuple[str | None, str | None]:
        if self._models_discovered or self.models:
            available = self._provider_models.get(provider, [])
            if not available:
                return None, None
            model = (
                self.resolve_model(preferred_model, provider)
                if preferred_model
                else None
            )
            effort_pref = preferred_effort
            if model is None and self.session is not None:
                def_model, def_effort, def_prov = self._workspace_defaults(self.session)
                if def_model is not None and def_prov == provider:
                    model = def_model
                    if effort_pref is None:
                        effort_pref = def_effort
            if model is None:
                model = self._provider_default_models.get(provider) or available[0]
            return model, self._select_effort(model, effort_pref)
        if preferred_model and (
            not self.model_providers
            or self.model_providers.get(preferred_model) == provider
        ):
            return preferred_model, preferred_effort
        if (
            self.session is not None
            and self.session.model is not None
            and self.session.provider_for_model(self.session.model) == provider
        ):
            return self.session.model, preferred_effort or self.session.effort
        if self.model is not None and (
            self.session is None
            or self.session.provider_for_model(self.model) == provider
        ):
            return self.model, preferred_effort or self.effort
        return None, None

    def _closed_session_meta(
        self, workspace, session_id: str
    ) -> tuple[str | None, str | None, str | None]:
        with suppress(Exception):
            path = workspace.transcript_file(session_id)
            stat = path.stat()
            cached = self._closed_session_meta_cache.get(path)
            if (
                cached is not None
                and cached[0] == stat.st_mtime_ns
                and cached[1] == stat.st_size
            ):
                return cached[2]
            records = workspace.read_transcript(path)
            meta = workspace.session_provider_and_model(records)
            self._closed_session_meta_cache[path] = (
                stat.st_mtime_ns,
                stat.st_size,
                meta,
            )
            return meta
        return None, None, None

    def _slurm_review_model(
        self,
        workspace,
        session_id: str,
        session: ExplorationSession | None = None,
    ) -> tuple[str | None, str | None, ProviderId | None]:
        owner_provider: ProviderId | None = None
        preferred_model: str | None = None
        preferred_effort: str | None = None
        if session is not None:
            owner_provider = session._active_provider or session._owner_provider
            preferred_model = session._active_model
            preferred_effort = session._active_effort
        if owner_provider is None:
            if session is not None:
                records: list[dict[str, Any]] = []
                with suppress(Exception):
                    records = session.workspace.current_records()
                raw_provider, rec_model, rec_effort = (
                    workspace.session_provider_and_model(records)
                )
            else:
                raw_provider, rec_model, rec_effort = self._closed_session_meta(
                    workspace, session_id
                )
            if raw_provider in self.backends:
                owner_provider = raw_provider
                preferred_model = preferred_model or rec_model
                preferred_effort = preferred_effort or rec_effort
        if owner_provider is not None:
            if (
                preferred_model is None
                and session is not None
                and session.model is not None
                and session.provider_for_model(session.model) == owner_provider
            ):
                preferred_model = session.model
                preferred_effort = preferred_effort or session.effort
            model, effort = self._select_provider_model(
                owner_provider, preferred_model, preferred_effort
            )
            return model, effort, owner_provider
        target = session or self.session
        if target is None or target.model is None:
            return None, None, None
        return target.model, target.effort, None

    async def _emit_waiting_review_notice(
        self,
        workspace,
        probe_id: str,
        owner_provider: ProviderId,
        target_session: ExplorationSession | None = None,
        target_agent_id: str | None = None,
    ) -> None:
        notice_key = (workspace.path, probe_id, owner_provider)
        if notice_key in self._slurm_waiting_notices:
            return
        target = target_session or self.session
        if target is None:
            return
        resolved_agent_id = (
            target_agent_id if target_session is not None else self.primary_agent_id
        )
        self._slurm_waiting_notices.add(notice_key)
        title = PROVIDER_TITLES.get(owner_provider, owner_provider)
        text = (
            f"Slurm probe {probe_id} finished; automatic review is waiting "
            f"for {title} models to become available."
        )
        await target.emit("system_message", text=text)
        if resolved_agent_id not in self._agent_subscribers:
            await self.emit(
                "system_message",
                {"agent_id": resolved_agent_id, "text": text},
            )

    async def _poll_slurm_jobs(self):
        if not self._started or self.changing or self.session is None:
            return
        workspace = self.session.workspace
        jobs = await asyncio.to_thread(slurm.pending_jobs, workspace)
        for job in jobs:
            if not self._monitoring_workspace(workspace):
                return
            probe_id = job["probe_id"]
            try:
                metadata, _ = await asyncio.to_thread(slurm.check, workspace, probe_id)
            except Exception as exc:
                # A scheduler outage is not an outcome of the scientific probe.
                logger.warning("Could not check %s: %s", probe_id, exc)
                continue
            if not self._monitoring_workspace(workspace):
                return
            terminal = metadata["status"] in ("completed", "error")
            key = (workspace.path, probe_id)
            session_id = probe_id.split("/")[0]
            agent_id = self.find_agent_for_session(session_id)
            if (
                terminal
                and self._discovering_models == 0
                and self._creating_agents == 0
                and agent_id is None
                and key not in self._slurm_review_attempts
                and not self._changing_agents
                and not any(
                    agent.has_active_query() for agent in list(self.agents.values())
                )
            ):
                review_model, _, owner_provider = self._slurm_review_model(
                    workspace, session_id
                )
                if review_model is not None:
                    # Resume the owning chat, never repurpose the user's current chat.
                    agent_id = await self.transition(
                        partial(
                            self.create_agent,
                            workspace.path,
                            session_id=session_id,
                        )
                    )
                    await self.emit("agent_added", self.agent_snapshot(agent_id))
                elif owner_provider is not None and await asyncio.to_thread(
                    slurm.needs_review, workspace, probe_id
                ):
                    await self._emit_waiting_review_notice(
                        workspace, probe_id, owner_provider
                    )
            if agent_id is None:
                continue
            session = self.agents.get(agent_id)
            if not self._slurm_owner_is_current(workspace, agent_id, session, probe_id):
                continue
            if session._probe_operation_lock.locked():
                continue
            state = (agent_id, metadata["status"], metadata.get("slurm_state"))
            if self._slurm_seen.get(key) != state:
                if terminal:
                    await session.emit(
                        "probe_finish",
                        **slurm.result_payload(
                            workspace, probe_id, metadata, session.image_cap
                        ),
                    )
                else:
                    probe = (
                        workspace.sessions_path
                        / session_id
                        / "probes"
                        / probe_id.split("/")[1]
                    )
                    await session.emit(
                        "probe_update",
                        probe_id=probe_id,
                        backend="slurm",
                        expected=metadata["expected"],
                        code=(probe / "code.py").read_text(),
                        output=f"Slurm job {metadata['job_id']}: "
                        f"{metadata.get('slurm_state', metadata['status'])}.",
                        status=metadata["status"],
                        slurm_state=metadata.get("slurm_state"),
                    )
                # Publishing can yield to a user changing the explorer or workspace.
                # Reserve a review only if this is still the probe's original chat.
                if not self._slurm_owner_is_current(
                    workspace, agent_id, session, probe_id
                ):
                    continue
                self._slurm_seen[key] = state
            if (
                terminal
                and self._discovering_models == 0
                and key not in self._slurm_review_attempts
                and not session.has_active_query()
                and agent_id not in self._changing_agents
                and session._pending_verdict_probe_id in (None, probe_id)
            ):
                review_model, review_effort, owner_provider = self._slurm_review_model(
                    workspace, session_id, session
                )
                if review_model is not None:
                    session._current_task = asyncio.create_task(
                        self._review_slurm(
                            session,
                            probe_id,
                            key,
                            model=review_model,
                            effort=review_effort,
                        )
                    )
                elif owner_provider is not None and await asyncio.to_thread(
                    slurm.needs_review, workspace, probe_id
                ):
                    await self._emit_waiting_review_notice(
                        workspace, probe_id, owner_provider, session, agent_id
                    )

    async def _review_slurm(
        self,
        session,
        probe_id,
        key,
        *,
        model: str | None = None,
        effort: str | None = None,
    ):
        try:
            with slurm.review_lock(session.workspace, probe_id) as claimed:
                if not claimed or not slurm.needs_review(session.workspace, probe_id):
                    return
                # Do not repeatedly bill for a failed or interrupted review. The saved
                # unresolved result is retried when the workspace is next opened.
                self._slurm_review_attempts.add(key)
                for notice_key in [
                    item
                    for item in self._slurm_waiting_notices
                    if item[0] == session.workspace.path and item[1] == probe_id
                ]:
                    self._slurm_waiting_notices.discard(notice_key)
                await asyncio.to_thread(
                    slurm.set_review_pending, session.workspace, probe_id, True
                )
                session._slurm_review_probe_id = probe_id
                if model is not None:
                    session.bind_next_turn(model, effort)
                completed = await session.review_slurm(probe_id)
                reviewed = slurm.has_verdict(session.workspace, probe_id)
                if completed and reviewed:
                    await asyncio.to_thread(
                        slurm.set_review_pending, session.workspace, probe_id, False
                    )
                elif await asyncio.to_thread(
                    slurm.needs_review, session.workspace, probe_id
                ):
                    reason = (
                        "did not record a verdict"
                        if not reviewed
                        else "did not finish its correction"
                    )
                    await session.emit(
                        "error",
                        message=f"Automatic review of {probe_id} {reason}. "
                        "The review remains unresolved; continue this chat to finish it, "
                        "or it will be retried when the workspace is reopened.",
                    )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Slurm review failed for %s", probe_id)
            await session.emit("error", message=f"Review of {probe_id} failed: {exc}")
        finally:
            session._bound_turn = None
            session._slurm_review_probe_id = None
            if session._current_task is asyncio.current_task():
                if session._finishing_task is session._current_task:
                    session._finishing_task = None
                session._current_task = None
            if (
                self._started
                and not self.changing
                and session._pending_prompts
                and session._current_task is None
            ):
                agent_id = next(
                    (
                        aid
                        for aid, candidate in self.agents.items()
                        if candidate is session
                    ),
                    None,
                )
                session._current_task = asyncio.create_task(
                    self._drain_pending_queries(session, agent_id)
                )
            if key in self._slurm_review_attempts:
                self._wake_slurm_monitor()

    async def discover_models(self):
        """Read model catalogues concurrently across all available provider CLIs."""
        self._discovering_models += 1
        self._discovery_seq += 1
        run_seq = self._discovery_seq
        auth_rev = self._auth_revision
        auth_snapshot = {provider: self.auth.get(provider) for provider in PROVIDERS}
        try:
            results = await asyncio.gather(
                *(
                    self.backends[provider].discover_models_staged(
                        auth_snapshot[provider]
                    )
                    for provider in PROVIDERS
                ),
                return_exceptions=True,
            )
            if (
                auth_rev != self._auth_revision
                or run_seq < self._published_discovery_seq
            ):
                return
            self._published_discovery_seq = run_seq

            models: list[str] = []
            model_labels: dict[str, str] = {}
            model_providers: dict[str, ProviderId] = {}
            model_raw_ids: dict[str, str] = {}
            model_efforts: dict[str, list[str]] = {}
            provider_models: dict[ProviderId, list[str]] = {p: [] for p in PROVIDERS}
            provider_default_models: dict[ProviderId, str] = {}
            provider_defaults: list[str] = []
            seen: set[str] = set()

            for provider, result in zip(PROVIDERS, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning("Could not discover %s models: %s", provider, result)
                    continue
                specs, commit = result
                try:
                    commit()
                except Exception as exc:
                    logger.warning("Could not discover %s models: %s", provider, exc)
                    continue
                for spec in specs:
                    spec_id = spec.id
                    if spec_id in seen:
                        spec_id = f"{spec.provider}/{spec.raw_id}"
                    if spec_id in seen:
                        continue
                    seen.add(spec_id)
                    models.append(spec_id)
                    model_labels[spec_id] = spec.label
                    model_providers[spec_id] = spec.provider
                    model_raw_ids[spec_id] = spec.raw_id
                    model_efforts[spec_id] = list(spec.efforts)
                    provider_models[spec.provider].append(spec_id)
                    if spec.is_default:
                        provider_defaults.append(spec_id)
                        provider_default_models.setdefault(spec.provider, spec_id)

            previous_providers = dict(self.model_providers)
            previous_raw_ids = dict(self.model_raw_ids)

            def previous_selection(
                model_id: str | None, fallback_provider: ProviderId | None = None
            ) -> tuple[str | None, ProviderId | None]:
                if model_id and model_id in previous_providers:
                    return previous_raw_ids.get(model_id, model_id), previous_providers[
                        model_id
                    ]
                return model_id, fallback_provider

            self.models = models
            self.model_labels = model_labels
            self.model_providers = model_providers
            self.model_raw_ids = model_raw_ids
            self.model_efforts = model_efforts
            self._provider_models = provider_models
            self._provider_default_models = provider_default_models
            self._models_discovered = True
            self.sdk_default_model = (
                provider_defaults[0]
                if provider_defaults
                else models[0]
                if models
                else None
            )
            if self.model is not None:
                model_pref, model_prov = previous_selection(self.model)
                self.model = self._select_model(model_pref, model_prov)
            else:
                self.model = self._select_model(self.startup_default_model)
            self.effort = self._select_effort(self.model, self.effort)
            if self.agents:
                for session in self.agents.values():
                    session.model_providers = dict(self.model_providers)
                    session.model_raw_ids = dict(self.model_raw_ids)
                if self._workspace_defaults_pending:
                    self._apply_workspace_defaults()
                else:
                    for session in self.agents.values():
                        default_model, default_effort, default_provider = (
                            self._workspace_defaults(session)
                        )
                        session.default_model = default_model
                        session.default_provider = default_provider
                        session.default_effort = default_effort
                        resolved_current: str | None = None
                        if session.model is not None:
                            sess_pref, sess_prov = previous_selection(session.model)
                            if sess_prov is not None:
                                resolved_current = self.resolve_model(
                                    sess_pref, sess_prov
                                )
                                if resolved_current is None:
                                    prov_models = self._provider_models.get(
                                        sess_prov, []
                                    )
                                    if prov_models:
                                        resolved_current = (
                                            self._provider_default_models.get(sess_prov)
                                            or prov_models[0]
                                        )
                            else:
                                resolved_current = self.resolve_model(sess_pref, None)
                        if resolved_current is not None:
                            session.model = resolved_current
                            session.effort = self._select_effort(
                                session.model, session.effort
                            )
                        else:
                            owner_provider = (
                                session._active_provider or session._owner_provider
                            )
                            owner_model, owner_effort = (
                                self._select_provider_model(
                                    owner_provider,
                                    session._active_model,
                                    session._active_effort,
                                )
                                if owner_provider is not None
                                else (None, None)
                            )
                            if owner_model is not None:
                                session.model = owner_model
                                session.effort = owner_effort
                            else:
                                session.model = default_model
                                session.effort = default_effort
                    if self.session is not None:
                        self.model = self.session.model
                        self.effort = self.session.effort
        finally:
            self._discovering_models = max(0, self._discovering_models - 1)
        self._wake_slurm_monitor()

    def resolve_model(
        self, preferred: str | None, preferred_provider: ProviderId | None = None
    ) -> str | None:
        if not preferred or not self.models:
            return None
        explicit_provider: ProviderId | None = None
        explicit_raw = preferred
        if "/" in preferred:
            prefix, _, rest = preferred.partition("/")
            if prefix in PROVIDERS:
                explicit_provider = prefix
                explicit_raw = rest
        target_provider = preferred_provider or explicit_provider
        target_raw = explicit_raw if explicit_provider else preferred
        if target_provider is not None:
            if (
                preferred in self.models
                and self.model_providers.get(preferred) == target_provider
            ):
                return preferred
            for spec_id in self.models:
                if (
                    self.model_providers.get(spec_id) == target_provider
                    and self.model_raw_ids.get(spec_id) == target_raw
                ):
                    return spec_id
            return None
        return preferred if preferred in self.models else None

    def _select_model(
        self, preferred: str | None, preferred_provider: ProviderId | None = None
    ) -> str | None:
        if preferred_provider is not None:
            resolved = self.resolve_model(preferred, preferred_provider)
            if resolved is not None:
                return resolved
            provider_models = self._provider_models.get(preferred_provider, [])
            if provider_models:
                return (
                    self._provider_default_models.get(preferred_provider)
                    or provider_models[0]
                )
        return (
            self.resolve_model(preferred, preferred_provider) or self.sdk_default_model
        )

    def _select_effort(self, model: str | None, preferred: str | None) -> str | None:
        levels = self.model_efforts.get(model or "", [])
        if preferred in levels:
            return preferred
        if DEFAULT_EFFORT in levels:
            return DEFAULT_EFFORT
        return levels[0] if levels else None

    def _apply_workspace_defaults(self):
        default_model, default_effort, default_provider = self._workspace_defaults(
            self.require_session()
        )
        for session in self.agents.values():
            self._apply_defaults_to(
                session, default_model, default_effort, default_provider
            )
            self._apply_resumed_owner_model(session)
        if self.session is not None:
            self.model = self.session.model
            self.effort = self.session.effort
        else:
            self.model = default_model
            self.effort = default_effort
        self._workspace_defaults_pending = False

    @staticmethod
    def _apply_defaults_to(
        session: ExplorationSession,
        model: str | None,
        effort: str | None,
        provider: ProviderId | None = None,
    ):
        session.default_model = model
        session.default_effort = effort
        session.default_provider = provider
        session.model = model
        session.effort = effort

    def _workspace_defaults(
        self, session: ExplorationSession
    ) -> tuple[str | None, str | None, ProviderId | None]:
        config = session.workspace.read_config()
        configured_model = config.get("default_model")
        if isinstance(configured_model, str):
            model_preference: str | None = configured_model
            configured_provider = config.get("default_provider")
            provider_preference: ProviderId | None = (
                configured_provider if configured_provider in PROVIDERS else None
            )
        else:
            model_preference = self.startup_default_model
            provider_preference = None
        model = self._select_model(model_preference, provider_preference)
        effort = self._select_effort(
            model,
            config.get("default_effort")
            if isinstance(config.get("default_effort"), str)
            else self.startup_default_effort,
        )
        return model, effort, self.model_providers.get(model or "")

    def _resolve_requested_model(
        self,
        model: str | None,
        *,
        provider: ProviderId | None = None,
        raw_model: str | None = None,
    ) -> str | None:
        if model is None:
            if raw_model is not None:
                raise ValueError("Choose a model from the model selector.")
            return None
        if provider is None:
            if raw_model is None:
                return model
            if not self._models_discovered and not self.model_providers:
                return model
            if model not in self.models:
                if self._models_discovered and not self.models:
                    raise ValueError("No model available.")
                raise ValueError("Choose a model from the model selector.")
            if self.model_raw_ids.get(model, model) != raw_model:
                raise ValueError("Choose a model from the model selector.")
            return model
        if provider not in PROVIDERS:
            raise ValueError("Choose a model from the model selector.")
        if not self._models_discovered and not self.model_providers:
            return model
        resolved = self.resolve_model(raw_model or model, provider)
        if resolved is None:
            if self._models_discovered and not self.models:
                raise ValueError("No model available.")
            raise ValueError("Choose a model from the model selector.")
        return resolved

    def set_session_defaults(
        self,
        model: str,
        effort: str | None,
        *,
        provider: ProviderId | None = None,
        raw_model: str | None = None,
    ):
        session = self.require_session()
        resolved_model = self._resolve_requested_model(
            model, provider=provider, raw_model=raw_model
        )
        if resolved_model is None or resolved_model not in self.models:
            raise ValueError("Choose a model from the model selector.")
        supported = self.model_efforts.get(resolved_model, [])
        if effort is not None and effort not in supported:
            raise ValueError("Choose an effort supported by the selected model.")
        if supported and effort is None:
            raise ValueError("Choose an effort supported by the selected model.")
        resolved_provider = self.model_providers.get(resolved_model)
        session.set_session_defaults(resolved_model, effort, provider=resolved_provider)
        for other in self.agents.values():
            other.default_model = resolved_model
            other.default_effort = effort
            other.default_provider = resolved_provider

    def reset_session(self):
        """Start a session with the workspace's configured defaults."""
        session = self.require_session()
        if self.models or self._models_discovered:
            (
                session.default_model,
                session.default_effort,
                session.default_provider,
            ) = self._workspace_defaults(session)
        session.reset_client_session()
        self.model = session.model
        self.effort = session.effort

    def resume_session(
        self, session_id: str, restore_kernel: bool = False
    ) -> dict[str, Any]:
        """Resume a recorded chat onto the active explorer and restore its owner model."""
        session = self.require_session()
        existing = self.find_agent_for_session(session_id)
        if existing is not None and self.agents.get(existing) is not session:
            raise RuntimeError(
                f"Session {session_id} is already open in another explorer."
            )
        data = session.resume_session(session_id, restore_kernel)
        if self.models or self._models_discovered:
            default_model, default_effort, default_provider = self._workspace_defaults(
                session
            )
            self._apply_defaults_to(
                session, default_model, default_effort, default_provider
            )
        self._apply_resumed_owner_model(session)
        self.model = session.model
        self.effort = session.effort
        return {**self.agent_snapshot(self.primary_agent_id), **data}

    async def authenticate(
        self, provider: ProviderId, method: str, credential: str | None = None
    ) -> dict[str, Any]:
        """Select the credential for one provider backend.

        Nothing is verified here: the provider CLI reports a bad credential on
        the next message. The model catalogue is refreshed as a courtesy.
        """
        self.require_all_idle("change authentication")
        self.auth.set(provider, method, credential)
        self._auth_revision += 1
        await self.discover_models()
        return self.auth.public(self._provider_models)

    def shutdown(self):
        self._started = False
        if self._slurm_task is not None:
            self._slurm_task.cancel()
        self.changing = False
        for agent_id in list(self.agents):
            self._unsubscribe_agent(agent_id)
            self.agents[agent_id].shutdown()

    def open_workspace(self, path: Path | str, python=None) -> ExplorationSession:
        """Open an existing directory, creating runtime state only after validation."""
        if self.agents:
            self.require_all_idle("change workspace")
        target = Path(path).expanduser().resolve()
        if not target.is_dir():
            raise ValueError("Choose an existing folder for the workspace.")

        # Keep the terminal's long-standing in-place session contract when it
        # is the only explorer. Multi-tab workspace changes intentionally tear
        # down every independent kernel below.
        if len(self.agents) == 1:
            session = self.require_session()
            session.set_workspace(target, python)
            self._slurm_seen.clear()
            self._slurm_review_attempts.clear()
            self._slurm_waiting_notices.clear()
            self._closed_session_meta_cache.clear()
            self._workspace_defaults_pending = True
            if self.models or self._models_discovered:
                self._apply_workspace_defaults()
            self._wake_slurm_monitor()
            return session

        new_session = self.session_factory(
            workspace_path=target,
            python=python,
            model=self.model,
            effort=self.effort,
            timeout_s=self.timeout_s,
            image_cap=self.image_cap,
            auth=self._auth,
            backends=self.backends,
            model_providers=self.model_providers,
            model_raw_ids=self.model_raw_ids,
        )
        with self._agents_lock:
            old_sessions = [self.agents[aid] for aid in list(self.agents)]
            for agent_id in list(self.agents):
                self._unsubscribe_agent(agent_id)
            self.agents.clear()
            self._slurm_seen.clear()
            self._slurm_review_attempts.clear()
            self._slurm_waiting_notices.clear()
            self._closed_session_meta_cache.clear()
            new_session.model_providers = dict(self.model_providers)
            new_session.model_raw_ids = dict(self.model_raw_ids)
            agent_id = self._new_agent_id()
            self.agents[agent_id] = new_session
            self.primary_agent_id = agent_id
            self.session = new_session
            self._workspace_defaults_pending = True
            if self.models or self._models_discovered:
                self._apply_workspace_defaults()
            self._subscribe_to_agent(agent_id, new_session)
        for old_session in old_sessions:
            old_session.shutdown()
        self._wake_slurm_monitor()
        return new_session

    def _new_agent_id(self) -> str:
        return uuid.uuid4().hex

    def _subscribe_to_agent(self, agent_id: str, session: ExplorationSession):
        if not self._started or agent_id in self._agent_subscribers:
            return

        async def forward(event_type: str, payload: dict[str, Any]):
            if event_type == "record_changed":
                for other_id, other in list(self.agents.items()):
                    if other_id != agent_id:
                        other.mark_record_dirty()
            await self.emit(event_type, {"agent_id": agent_id, **payload})
            if (
                event_type == "probe_verdict"
                or (
                    event_type == "probe_update"
                    and payload.get("status") == "submitted"
                )
                or (event_type == "status_change" and payload.get("status") == "idle")
            ):
                self._wake_slurm_monitor()

        session.subscribe(forward)
        self._agent_subscribers[agent_id] = forward

    def _unsubscribe_agent(self, agent_id: str):
        session = self.agents.get(agent_id)
        callback = self._agent_subscribers.pop(agent_id, None)
        if session and callback:
            session.unsubscribe(callback)

    def subscribe(self, callback: Callable[[str, dict[str, Any]], Any]):
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, dict[str, Any]], Any]):
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    async def emit(self, event_type: str, data: dict[str, Any] | None = None):
        payload = data or {}
        for subscriber in list(self.subscribers):
            try:
                result = subscriber(event_type, payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error(f"Error in runtime subscriber: {exc}")

    def _apply_resumed_owner_model(self, session: ExplorationSession) -> None:
        owner_provider = session._active_provider or session._owner_provider
        if owner_provider is None:
            return
        owner_model, owner_effort = self._select_provider_model(
            owner_provider,
            session._active_model,
            session._active_effort,
        )
        if owner_model is not None:
            session.model = owner_model
            session.effort = owner_effort

    def create_agent(
        self,
        workspace_path: Path | str | None = None,
        *,
        python: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Create an independent exploration in the current workspace."""
        if workspace_path is None:
            workspace_path = self.require_session().workspace_path
        session = self.session_factory(
            workspace_path=workspace_path,
            python=python,
            model=self.model,
            effort=self.effort,
            timeout_s=self.timeout_s,
            image_cap=self.image_cap,
            auth=self._auth,
            backends=self.backends,
            model_providers=self.model_providers,
            model_raw_ids=self.model_raw_ids,
        )
        if session_id is not None:
            try:
                session.resume_session(session_id)
            except Exception:
                session.shutdown()
                raise
        session.model_providers = dict(self.model_providers)
        session.model_raw_ids = dict(self.model_raw_ids)
        agent_id = self._new_agent_id()
        with self._agents_lock:
            self.agents[agent_id] = session
            if self.primary_agent_id is None:
                self.primary_agent_id = agent_id
                self.session = session
            if self.models or self._models_discovered:
                default_model, default_effort, default_provider = (
                    self._workspace_defaults(session)
                )
                self._apply_defaults_to(
                    session, default_model, default_effort, default_provider
                )
                if session_id is not None:
                    self._apply_resumed_owner_model(session)
            elif session_id is not None:
                owner_provider = session._active_provider or session._owner_provider
                if owner_provider is not None and (
                    session.model is None
                    or session.provider_for_model(session.model) != owner_provider
                ):
                    self._apply_resumed_owner_model(session)
            self._subscribe_to_agent(agent_id, session)
        return agent_id

    async def create_agent_async(self) -> str:
        """Create a new explorer asynchronously while guarding concurrent transitions."""
        if self.changing:
            raise RuntimeError(
                "The research environment is being prepared. Please wait."
            )
        target_workspace = self.require_session().workspace_path
        self._creating_agents += 1
        task = asyncio.create_task(
            asyncio.to_thread(self.create_agent, target_workspace)
        )
        try:
            agent_id = await asyncio.shield(task)
        except asyncio.CancelledError:
            agent_id = await task
            if not self._started:
                self._unsubscribe_agent(agent_id)
                created = self.agents.pop(agent_id, None)
                if created is not None:
                    created.shutdown()
            raise
        finally:
            self._creating_agents = max(0, self._creating_agents - 1)
        if not self._started and self.agents.get(agent_id) is not None:
            self._unsubscribe_agent(agent_id)
            created = self.agents.pop(agent_id, None)
            if created is not None:
                created.shutdown()
            raise RuntimeError("Runtime has stopped.")
        created_session = self.agents.get(agent_id)
        if created_session is not None:
            created_session.model_providers = dict(self.model_providers)
            created_session.model_raw_ids = dict(self.model_raw_ids)
        return agent_id

    def get_agent(self, agent_id: str | None = None) -> ExplorationSession:
        if agent_id is None:
            return self.require_session()
        try:
            return self.agents[agent_id]
        except KeyError as exc:
            raise ValueError("Explorer not found.") from exc

    def agent_snapshot(self, agent_id: str) -> dict[str, Any]:
        session = self.get_agent(agent_id)
        return {
            "agent_id": agent_id,
            "session_id": None
            if session.workspace.session_pending
            else session.workspace.session_id,
            "records": session.workspace.current_records(),
            "model": session.model,
            "effort": session.effort,
            "can_compact": session.can_compact,
            "context_usage": session.context_usage,
            "kernel_alive": session.kernel.is_alive(),
            "kernel_busy": session.kernel.busy,
            "is_busy": session.has_active_query() or agent_id in self._changing_agents,
            "status": session.status,
            "pending_prompts": session.pending_prompts_snapshot(),
            "documents": session.list_documents(),
            "folders": session.workspace.list_document_folders(),
            "documents_default": session.documents_default,
        }

    def close_agent(self, agent_id: str):
        with self._agents_lock:
            if len(self.agents) <= 1:
                raise RuntimeError("Keep one explorer open.")
            session = self.get_agent(agent_id)
            session.require_idle("close this explorer")
            self._unsubscribe_agent(agent_id)
            del self.agents[agent_id]
            if self.primary_agent_id == agent_id:
                self.primary_agent_id = next(iter(self.agents))
                self.session = self.agents[self.primary_agent_id]
                self.model = self.session.model
                self.effort = self.session.effort
        session.shutdown()

    def find_agent_for_session(self, session_id: str) -> str | None:
        for agent_id, session in list(self.agents.items()):
            if (
                not session.workspace.session_pending
                and session.workspace.session_id == session_id
            ):
                return agent_id
        return None

    def replace_agent(
        self, agent_id: str, session_id: str, restore_kernel: bool = False
    ) -> tuple[str, dict[str, Any], bool]:
        """Load history into an idle tab, unless it is already open elsewhere."""
        with self._agents_lock:
            existing = self.find_agent_for_session(session_id)
            if existing is not None and existing != agent_id:
                return existing, self.agent_snapshot(existing), True
            if session_id in self._resuming_sessions:
                raise RuntimeError(
                    f"Session {session_id} is already being resumed in another explorer."
                )
            old = self.get_agent(agent_id)
            old.require_idle("resume a past session")
            self._resuming_sessions.add(session_id)
        try:
            replacement = self.session_factory(
                workspace_path=old.workspace_path,
                model=old.model,
                effort=old.effort,
                timeout_s=old.timeout_s,
                image_cap=old.image_cap,
                auth=self._auth,
                backends=self.backends,
                model_providers=self.model_providers,
                model_raw_ids=self.model_raw_ids,
            )
            replacement.default_model = old.default_model
            replacement.default_effort = old.default_effort
            replacement.default_provider = old.default_provider
            try:
                data = replacement.resume_session(session_id, restore_kernel)
            except Exception:
                replacement.shutdown()
                raise
            with self._agents_lock:
                replacement.model_providers = dict(self.model_providers)
                replacement.model_raw_ids = dict(self.model_raw_ids)
                if self.models or self._models_discovered:
                    default_model, default_effort, default_provider = (
                        self._workspace_defaults(replacement)
                    )
                    self._apply_defaults_to(
                        replacement, default_model, default_effort, default_provider
                    )
                self._apply_resumed_owner_model(replacement)
                existing = self.find_agent_for_session(session_id)
                if existing is not None and existing != agent_id:
                    discard = replacement
                    snapshot = self.agent_snapshot(existing)
                    focused_id = existing
                else:
                    discard = None
                    self._unsubscribe_agent(agent_id)
                    self.agents[agent_id] = replacement
                    if self.primary_agent_id == agent_id:
                        self.session = replacement
                        self.model = replacement.model
                        self.effort = replacement.effort
                    self._subscribe_to_agent(agent_id, replacement)
                    snapshot = {**self.agent_snapshot(agent_id), **data}
                    focused_id = agent_id
            if discard is not None:
                discard.shutdown()
                return focused_id, snapshot, True
            old.shutdown()
            return focused_id, snapshot, False
        finally:
            with self._agents_lock:
                self._resuming_sessions.discard(session_id)

    def workspace_snapshot(self) -> dict[str, Any]:
        """Describe the current workspace for a client that just connected."""
        session = self.session
        auth_public = self.auth.public(self._provider_models)
        if session is None:
            return {
                "is_open": False,
                "path": None,
                "model": self.model,
                "effort": self.effort,
                "models": self.models,
                "model_labels": self.model_labels,
                "model_providers": self.model_providers,
                "model_raw_ids": self.model_raw_ids,
                "model_efforts": self.model_efforts,
                "session_defaults": None,
                "prompt_mode": DEFAULT_PROMPT_MODE,
                "evidence_path": EVIDENCE_PATH,
                "thoughts_path": THOUGHTS_PATH,
                "plots": [],
                "agents": [],
                "navigator_root": str(self.navigator_root),
                "auth": auth_public,
            }
        return {
            "is_open": True,
            "path": str(session.workspace_path),
            # Legacy single-explorer fields keep the terminal-adjacent HTTP
            # surface useful while web clients consume ``agents`` below.
            "session_id": None
            if session.workspace.session_pending
            else session.workspace.session_id,
            "records": session.workspace.current_records(),
            "environment": {
                "python": str(session.environment.python),
                "version": session.environment.version,
            },
            "model": session.model,
            "effort": session.effort,
            "models": self.models,
            "model_labels": self.model_labels,
            "model_providers": self.model_providers,
            "model_raw_ids": self.model_raw_ids,
            "model_efforts": self.model_efforts,
            "session_defaults": {
                "model": session.default_model,
                "effort": session.default_effort,
            },
            "exploration_prompt": session.exploration_prompt,
            "exploration_prompt_is_custom": session.exploration_prompt_override
            is not None,
            "prompt_mode": session.prompt_mode,
            "evidence_path": EVIDENCE_PATH,
            "thoughts_path": THOUGHTS_PATH,
            "kernel_alive": session.kernel.is_alive(),
            "kernel_busy": session.kernel.busy,
            "is_busy": session.has_active_query(),
            "context_usage": session.context_usage,
            "documents": session.list_documents(),
            "folders": session.workspace.list_document_folders(),
            "documents_default": session.documents_default,
            "plots": session.list_plots(),
            "agents": [
                self.agent_snapshot(agent_id)
                for agent_id in list(self.agents)
                if agent_id in self.agents
            ],
            "auth": auth_public,
        }

    def export_notebook(
        self, agent_id: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Render the active session as a notebook and the filename to save it under."""
        session = self.get_agent(agent_id)
        if session.workspace.session_pending:
            raise ValueError("The current session has no messages to download.")
        notebook = session.workspace.export_notebook(
            session.workspace.session_id, session.environment.version
        )
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", session.workspace_path.name).strip(".-")
        filename = (
            f"nocturnomath-{name or 'workspace'}-{session.workspace.session_id}.ipynb"
        )
        return filename, notebook

    def require_session(self) -> ExplorationSession:
        if self.session is None:
            raise RuntimeError("Open a workspace folder before using the agent.")
        return self.session

    def require_all_idle(self, action: str):
        if self._changing_agents or self._creating_agents:
            raise RuntimeError(f"Cannot {action} while an explorer is changing.")
        for session in list(self.agents.values()):
            session.require_idle(action)

    async def transition(self, operation, *args):
        if self.changing:
            raise RuntimeError(
                "The research environment is being prepared. Please wait."
            )
        self.require_all_idle("change kernel or workspace")
        self.changing = True
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        finally:
            self.changing = False
            self._wake_slurm_monitor()

    async def transition_agent(self, agent_id: str, operation, *args):
        if self.changing:
            raise RuntimeError(
                "The research environment is being prepared. Please wait."
            )
        if agent_id in self._changing_agents:
            raise RuntimeError("This explorer is already changing. Please wait.")
        self.get_agent(agent_id).require_idle("change this explorer")
        self._changing_agents.add(agent_id)
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        finally:
            self._changing_agents.discard(agent_id)
            self._wake_slurm_monitor()

    async def compact_agent(self, agent_id: str):
        if self.changing:
            raise RuntimeError(
                "The research environment is being prepared. Please wait."
            )
        if agent_id in self._changing_agents:
            raise RuntimeError("This explorer is already changing. Please wait.")
        session = self.get_agent(agent_id)
        session.require_idle("compact this explorer")
        session._compaction_reserved = True
        session._is_busy = True
        session.status = "compacting"
        compact_epoch = session._interrupt_epoch
        task = asyncio.create_task(session.compact())
        session._current_task = task
        succeeded = False
        outer = asyncio.current_task()
        try:
            result = await asyncio.shield(task)
            succeeded = True
            return result
        except asyncio.CancelledError:
            outer_cancelled = outer is not None and bool(outer.cancelling())
            try:
                result = await task
                succeeded = True
            except asyncio.CancelledError:
                await session.clear_pending_prompts(through_epoch=compact_epoch)
                if outer_cancelled:
                    raise
                raise RuntimeError("Compaction was cancelled.") from None
            except Exception:
                await session.clear_pending_prompts(through_epoch=compact_epoch)
                raise
            if outer_cancelled:
                raise
            return result
        except Exception:
            await session.clear_pending_prompts(through_epoch=compact_epoch)
            raise
        finally:
            if session._current_task is task:
                session._current_task = None
            session._compaction_reserved = False
            session._is_busy = False
            if session.status == "compacting":
                session.status = "idle"
            if (
                self._started
                and not self.changing
                and session._pending_prompts
                and session._current_task is None
            ):
                if succeeded or not any(
                    item.epoch <= compact_epoch for item in session._pending_prompts
                ):
                    session._current_task = asyncio.create_task(
                        self._drain_pending_queries(session, agent_id)
                    )
                else:
                    await session.clear_pending_prompts()
            self._wake_slurm_monitor()

    async def set_prompt_mode(self, mode: str) -> str:
        """Persist workspace prompt delivery mode and update any still-pending prompts."""
        session = self.require_session()
        resolved = session.set_prompt_mode(mode)
        for other in list(self.agents.values()):
            if (
                other.workspace_path == session.workspace_path
                and other._pending_prompts
            ):
                for item in other._pending_prompts:
                    item.mode = resolved
                await other.emit(
                    "pending_prompts_changed",
                    pending_prompts=other.pending_prompts_snapshot(),
                )
        return resolved

    def _validate_query_model(
        self,
        session: ExplorationSession,
        model: str | None = None,
        effort: str | None = None,
    ):
        if model is None and session.model is None:
            raise ValueError("No model available.")
        if model is not None and model not in self.models:
            raise ValueError("Choose a model from the model selector.")
        target_model = model or session.model
        if self._models_discovered and target_model not in self.models:
            if not self.models:
                raise ValueError("No model available.")
            raise ValueError("Choose a model from the model selector.")
        supported_efforts = self.model_efforts.get(target_model or "", [])
        if effort is not None and effort not in supported_efforts:
            raise ValueError("Choose an effort supported by the selected model.")

    def _apply_query_model(
        self,
        session: ExplorationSession,
        model: str | None = None,
        effort: str | None = None,
        agent_id: str | None = None,
    ):
        metadata = {}
        if model is not None:
            session.model = model
            if agent_id is None or agent_id == self.primary_agent_id:
                self.model = model
            metadata["model"] = model
            selected_effort = self._select_effort(model, effort or session.effort)
            session.effort = selected_effort
            if agent_id is None or agent_id == self.primary_agent_id:
                self.effort = selected_effort
            metadata["effort"] = selected_effort
        elif effort is not None:
            session.effort = effort
            if agent_id is None or agent_id == self.primary_agent_id:
                self.effort = effort
            metadata["effort"] = effort
        if metadata:
            session.log_transcript("meta", **metadata)

    def start_query(
        self,
        text: str,
        model: str | None = None,
        effort: str | None = None,
        agent_id: str | None = None,
        *,
        provider: ProviderId | None = None,
        raw_model: str | None = None,
    ) -> str:
        if self.changing:
            raise RuntimeError(
                "The research environment is being prepared. Please wait."
            )
        changing_id = agent_id if agent_id is not None else self.primary_agent_id
        session = self.get_agent(agent_id)
        if changing_id in self._changing_agents and session.status != "compacting":
            raise RuntimeError("This explorer is changing. Please wait.")
        resolved_model = self._resolve_requested_model(
            model, provider=provider, raw_model=raw_model
        )
        self._validate_query_model(session, resolved_model, effort)
        if session.has_active_query():
            session.enqueue_prompt(
                text, mode=session.prompt_mode, model=resolved_model, effort=effort
            )
            if (
                not session._is_busy
                and changing_id not in self._changing_agents
                and (session._current_task is None or session._current_task.done())
            ):
                session._current_task = asyncio.create_task(
                    self._wait_and_drain_pending(session, agent_id)
                )
            return "pending"

        self._apply_query_model(session, resolved_model, effort, agent_id)
        session.bind_next_turn()
        # An accepted first message allocates its session before the UI sees it.
        session.activate_session()
        task = asyncio.create_task(self._run_query(session, text, agent_id))
        session._current_task = task
        return "started"

    @property
    def current_task(self) -> asyncio.Task | None:
        return self.session._current_task if self.session else None

    async def wait_for_query(self):
        """Wait for the running query without re-raising its cancellation."""
        task = self.current_task
        if task:
            await asyncio.gather(task, return_exceptions=True)

    async def drain(self):
        """Interrupt every explorer and preserve any partial probe artifacts."""
        self._started = False
        if self._slurm_task is not None:
            self._slurm_task.cancel()
            await asyncio.gather(self._slurm_task, return_exceptions=True)
        sessions = list(self.agents.values())
        await asyncio.gather(*(session.interrupt() for session in sessions))
        await asyncio.gather(
            *(
                asyncio.gather(
                    *(
                        task
                        for task in (session._current_task, session._probe_task)
                        if task is not None
                    ),
                    return_exceptions=True,
                )
                for session in sessions
            )
        )

    async def _drain_pending_loop(
        self, session: ExplorationSession, agent_id: str | None
    ) -> bool:
        while session._pending_prompts:
            batch = session.pop_next_turn_prompts()
            batch_model: str | None = None
            batch_effort: str | None = None
            batch_provider: ProviderId | None = None
            batch_raw_model: str | None = None
            for item in batch:
                if item.model is not None:
                    item_provider = item.provider or session.provider_for_model(
                        item.model
                    )
                    item_raw_model = item.raw_model or session.raw_model_id(item.model)
                    same_target = batch_model is not None and (
                        item_provider,
                        item_raw_model,
                    ) == (batch_provider, batch_raw_model)
                    batch_model = item.model
                    batch_provider = item_provider
                    batch_raw_model = item_raw_model
                    if item.effort is not None or not same_target:
                        batch_effort = item.effort
                elif item.effort is not None:
                    batch_effort = item.effort
            if (
                batch_model is not None
                and batch_provider is not None
                and self.model_providers
            ):
                resolved = self.resolve_model(
                    batch_raw_model or batch_model, batch_provider
                )
                batch_model = (
                    resolved
                    if resolved is not None
                    else f"{batch_provider}/{batch_raw_model or batch_model}"
                )
            try:
                self._validate_query_model(session, batch_model, batch_effort)
            except ValueError as exc:
                payload: dict[str, Any] = {"message": str(exc)}
                if agent_id is not None:
                    payload["agent_id"] = agent_id
                await self.emit("error", payload)
                await session.clear_pending_prompts()
                return False
            self._apply_query_model(session, batch_model, batch_effort, agent_id)
            session.bind_next_turn()
            session.activate_session()
            for item in batch:
                await session.emit(
                    "user_message",
                    session_id=session.workspace.session_id,
                    text=item.text,
                    model=session.model,
                    effort=session.effort,
                )
            await session.emit(
                "pending_prompts_changed",
                pending_prompts=session.pending_prompts_snapshot(),
            )
            session._pending_user_logs = (
                [item.text for item in batch] if len(batch) > 1 else None
            )
            try:
                ok = await session.query("\n\n".join(item.text for item in batch))
            finally:
                session._bound_turn = None
                session._pending_user_logs = None
            if ok is False:
                await session.clear_pending_prompts()
                return False
        return True

    async def _wait_and_drain_pending(
        self, session: ExplorationSession, agent_id: str | None
    ):
        current = asyncio.current_task()
        wait_epoch = session._interrupt_epoch
        try:
            while True:
                probe_task = session._probe_task
                if probe_task is not None:
                    if not probe_task.done():
                        try:
                            await asyncio.shield(probe_task)
                        except asyncio.CancelledError:
                            if current is not None and current.cancelling():
                                raise
                        except Exception as exc:
                            logger.debug(
                                "Background probe task ended with error: %s", exc
                            )
                    if session._probe_task is probe_task and probe_task.done():
                        session._probe_task = None
                    if probe_task is not session._probe_task:
                        continue
                if session._probe_operation_lock.locked():
                    async with session._probe_operation_lock:
                        pass
                    continue
                if session.kernel.busy:
                    await asyncio.sleep(0.02)
                    continue
                break
        except asyncio.CancelledError:
            await session.clear_pending_prompts(through_epoch=wait_epoch)
            if session._current_task is current:
                session._current_task = None
            if (
                self._started
                and not self.changing
                and session._pending_prompts
                and session._current_task is None
            ):
                session._current_task = asyncio.create_task(
                    self._drain_pending_queries(session, agent_id)
                )
            return
        await self._run_query_body(session, agent_id)

    async def _run_query_body(
        self,
        session: ExplorationSession,
        agent_id: str | None,
        initial_text: str | None = None,
    ):
        failed = False
        session._draining_queries = True
        try:
            ok = True
            if initial_text is not None:
                ok = await session.query(initial_text)
            if ok is False:
                failed = True
            else:
                while session._pending_prompts:
                    if not await self._drain_pending_loop(session, agent_id):
                        failed = True
                        break
                    if (
                        not session._pending_prompts
                        and not session._is_busy
                        and session.status != "idle"
                    ):
                        await session.emit("status_change", status="idle")
        except asyncio.CancelledError:
            if not self._started:
                failed = True
                await session.clear_pending_prompts()
        except Exception as exc:
            failed = True
            await session.clear_pending_prompts()
            logger.exception("Query task failed")
            payload = {"message": str(exc)}
            if agent_id is not None:
                payload["agent_id"] = agent_id
            await self.emit("error", payload)
        finally:
            session._bound_turn = None
            session._pending_user_logs = None
            session._draining_queries = False
            if not session._is_busy and session.status != "idle":
                await session.emit("status_change", status="idle")
            if session._current_task is asyncio.current_task():
                if session._finishing_task is session._current_task:
                    session._finishing_task = None
                session._current_task = None
            if session._pending_prompts:
                if not failed and not self.changing and session._current_task is None:
                    session._current_task = asyncio.create_task(
                        self._drain_pending_queries(session, agent_id)
                    )
                else:
                    await session.clear_pending_prompts()

    async def _run_query(
        self, session: ExplorationSession, text: str, agent_id: str | None
    ):
        await self._run_query_body(session, agent_id, text)

    async def _drain_pending_queries(
        self, session: ExplorationSession, agent_id: str | None
    ):
        await self._run_query_body(session, agent_id)
