"""FastAPI application factory for the Nocturnomath dashboard."""

import asyncio
import json
import logging
import webbrowser
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, SecretStr
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..auth import ProviderId
from ..backends import AgentBackend
from ..session import ExplorationSession
from .runtime import WebRuntime

logger = logging.getLogger("nocturnomath.web")
STATIC_DIR = Path(__file__).parent / "static"
SAFE_HOSTS = ["127.0.0.1", "localhost", "[::1]", "testserver"]


def _same_origin(origin: str, scheme: str, host: str) -> bool:
    """Compare an HTTP Origin header with the request endpoint."""
    try:
        parsed = urlsplit(origin)
        expected_scheme = "https" if scheme in ("https", "wss") else "http"
        return (
            parsed.scheme == expected_scheme
            and parsed.netloc == host
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


class WorkspaceChangeRequest(BaseModel):
    path: str
    python: str | None = None


class SessionResumeRequest(BaseModel):
    id: str
    restore_kernel: bool = False


class DocumentCreateRequest(BaseModel):
    title: str
    text: str = ""
    folder: str = ""


class DocumentFolderCreateRequest(BaseModel):
    path: str


class DocumentWriteRequest(BaseModel):
    text: str


class DocumentIncludeRequest(BaseModel):
    included: bool


class DocumentsDefaultRequest(BaseModel):
    enabled: bool


class SessionDefaultsRequest(BaseModel):
    model: str
    effort: str | None = None
    provider: Literal["anthropic", "openai", "google"] | None = None
    raw_model: str | None = None


class ExplorationPromptRequest(BaseModel):
    prompt: str | None


class PromptModeRequest(BaseModel):
    mode: str


class AuthRequest(BaseModel):
    provider: Literal["anthropic", "openai", "google"]
    method: Literal["cli", "subscription", "api_key"]
    credential: SecretStr | None = None


def create_app(
    session: ExplorationSession | None = None,
    *,
    browser_url: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    timeout_s: int = 600,
    image_cap: int = 2,
    navigator_root: Path | str = ".",
    python: str | None = None,
    session_factory: Callable[..., ExplorationSession] = ExplorationSession,
    backends: Mapping[ProviderId, AgentBackend] | None = None,
) -> FastAPI:
    """Build the dashboard without opening a workspace until the user chooses one."""
    runtime = WebRuntime(
        session,
        model=model,
        effort=effort,
        timeout_s=timeout_s,
        image_cap=image_cap,
        navigator_root=navigator_root,
        session_factory=session_factory,
        backends=backends,
    )

    initial_python = python
    interrupt_tasks: dict[str, asyncio.Task] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime.discover_models()
        runtime.start()
        browser_task = None
        if browser_url:

            async def open_browser_later():
                await asyncio.sleep(1.0)
                webbrowser.open(browser_url, new=2)

            browser_task = asyncio.create_task(open_browser_later())
        try:
            yield
        finally:
            if browser_task and not browser_task.done():
                browser_task.cancel()
            await runtime.drain()
            for task in list(interrupt_tasks.values()):
                if not task.done():
                    task.cancel()
            if interrupt_tasks:
                await asyncio.gather(*interrupt_tasks.values(), return_exceptions=True)
            runtime.shutdown()

    app = FastAPI(title="Nocturnomath Web App", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.request_shutdown = None
    app.state.exiting = False

    # SSH forwarding presents the app through localhost. Reject alternate Host
    # values so DNS rebinding cannot turn another web origin into this one.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=SAFE_HOSTS)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # FastAPI's default 422 body echoes the rejected input, which for
        # /api/auth would be a credential. Keep only the location and message.
        errors = [
            {key: value for key, value in error.items() if key not in ("input", "ctx")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.middleware("http")
    async def protect_browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if (
            request.method not in ("GET", "HEAD", "OPTIONS")
            and origin
            and not _same_origin(
                origin, request.url.scheme, request.headers.get("host", "")
            )
        ):
            return JSONResponse(
                status_code=403,
                content={"detail": "Changes must be requested from this app."},
            )
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith(
            ("/api/", "/static/")
        ):
            response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "font-src 'self' https://cdn.jsdelivr.net https://fonts.gstatic.com; "
            "img-src 'self' data: https:; connect-src 'self' ws: wss:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        return response

    @contextmanager
    def http_errors(
        *,
        value_status: int = 400,
        not_found: str | None = None,
        not_found_status: int = 404,
    ):
        try:
            yield
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=not_found_status, detail=not_found or str(exc)
            ) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=value_status, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def ws_error(message: str, agent_id: str | None = None) -> None:
        payload = {"message": message}
        if agent_id is not None:
            payload["agent_id"] = agent_id
        await runtime.broadcast("error", payload)

    def active_session(agent_id: str | None = None) -> ExplorationSession:
        if runtime.session is None:
            raise HTTPException(
                status_code=409,
                detail="Open a workspace folder before using the agent.",
            )
        with http_errors(value_status=404):
            return runtime.get_agent(agent_id)

    def require_idle(
        action: str, agent_id: str | None = None
    ) -> tuple[str | None, ExplorationSession]:
        session = active_session(agent_id)
        resolved_agent_id = agent_id or runtime.primary_agent_id
        if (
            runtime.changing
            or resolved_agent_id in runtime._changing_agents
            or session.has_active_query()
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot {action} while the agent is running a query.",
            )
        return resolved_agent_id, session

    def navigation_path(path: Path | str) -> Path:
        target = Path(path).expanduser().resolve()
        if not target.is_relative_to(runtime.navigator_root):
            raise PermissionError("Folder is outside the configured workspace root.")
        return target

    async def create_and_broadcast_agent(requested_by: str | None = None):
        new_id = await runtime.create_agent_async()
        snapshot = runtime.agent_snapshot(new_id)
        payload = (
            {"requested_by": requested_by, **snapshot}
            if requested_by is not None
            else snapshot
        )
        await runtime.broadcast("agent_added", payload)
        return snapshot

    async def restart_and_broadcast_kernel(agent_id: str, session: ExplorationSession):
        await runtime.transition_agent(agent_id, session.kernel.restart)
        alive = session.kernel.is_alive()
        await runtime.broadcast(
            "kernel_restarted", {"agent_id": agent_id, "kernel_alive": alive}
        )
        return alive

    async def open_and_broadcast_workspace(path: Path | str, python: str | None):
        nonlocal initial_python
        target = navigation_path(path)
        await runtime.transition(
            runtime.open_workspace, target, python or initial_python
        )
        initial_python = None
        info = runtime.workspace_snapshot()
        await runtime.broadcast("workspace_updated", {"workspace": info})
        return info

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_path)

    @app.get("/api/workspace")
    async def get_workspace_info():
        return runtime.workspace_snapshot()

    @app.get("/api/agents/{agent_id}")
    async def get_agent(agent_id: str):
        with http_errors(value_status=404):
            return runtime.agent_snapshot(agent_id)

    @app.post("/api/agents", status_code=201)
    async def create_agent():
        active_session()
        try:
            return await create_and_broadcast_agent()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to create explorer")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/agents/{agent_id}")
    async def close_agent(agent_id: str):
        with http_errors(value_status=404):
            await runtime.transition_agent(agent_id, runtime.close_agent, agent_id)
        await runtime.broadcast("agent_removed", {"agent_id": agent_id})
        return {"status": "ok"}

    @app.post("/api/agents/{agent_id}/resume")
    async def replace_agent(agent_id: str, request: SessionResumeRequest):
        with http_errors(value_status=404, not_found="Session not found"):
            resolved_id, data, focused = await runtime.transition_agent(
                agent_id,
                runtime.replace_agent,
                agent_id,
                request.id,
                request.restore_kernel,
            )
        event = "agent_focused" if focused else "session_resumed"
        payload = {"agent_id": resolved_id, **data}
        await runtime.broadcast(event, payload)
        return {"status": "ok", "focused_existing": focused, **payload}

    @app.post("/api/agents/{agent_id}/compact")
    async def compact_agent(agent_id: str):
        try:
            with http_errors(value_status=404):
                await runtime.compact_agent(agent_id)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to compact explorer")
            raise HTTPException(
                status_code=502, detail=f"Compaction failed: {exc}"
            ) from exc
        return {
            "status": "ok",
            "context_usage": runtime.get_agent(agent_id).context_usage,
        }

    def auth_payload():
        return {
            "auth": runtime.auth_public(),
            "model": runtime.current_model,
            **runtime.catalogue_payload(),
            "effort": runtime.current_effort,
            "agent_models": {
                agent_id: {"model": agent.model, "effort": agent.effort}
                for agent_id, agent in list(runtime.agents.items())
            },
            "session_defaults": (
                {
                    "model": runtime.session.default_model,
                    "effort": runtime.session.default_effort,
                }
                if runtime.session
                else None
            ),
        }

    @app.get("/api/auth")
    async def get_auth():
        return auth_payload()

    @app.post("/api/auth")
    async def set_auth(auth_request: AuthRequest):
        credential = (
            auth_request.credential.get_secret_value()
            if auth_request.credential is not None
            else None
        )
        with http_errors():
            await runtime.authenticate(
                auth_request.provider, auth_request.method, credential
            )
        data = auth_payload()
        await runtime.broadcast(
            "auth_changed", {"provider": auth_request.provider, **data}
        )
        return data

    @app.post("/api/exit")
    async def exit_service(background_tasks: BackgroundTasks):
        if app.state.request_shutdown is None:
            raise HTTPException(
                status_code=503, detail="Stop this service using its launcher."
            )
        if not app.state.exiting:
            app.state.exiting = True

            async def stop():
                await runtime.broadcast("service_stopping", {})
                await runtime.drain()
                app.state.request_shutdown()

            background_tasks.add_task(stop)
        return {"status": "stopping"}

    @app.post("/api/workspace")
    async def change_workspace(request: WorkspaceChangeRequest):
        if runtime.session is not None:
            require_idle("change workspace")
        try:
            with http_errors():
                info = await open_and_broadcast_workspace(request.path, request.python)
                return {"status": "ok", "workspace": info}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to change workspace")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/folders")
    async def list_folders(path: str | None = None):
        """List child directories for the visual workspace picker without writing."""
        requested = path or str(runtime.navigator_root)
        with http_errors():
            directory = navigation_path(requested)
        if not directory.is_dir():
            raise HTTPException(status_code=404, detail="Folder not found")
        try:
            folders = sorted(
                (
                    {"name": child.name, "path": str(child)}
                    for child in directory.iterdir()
                    if not child.name.startswith(".")
                    and child.is_dir()
                    and not child.is_symlink()
                ),
                key=lambda item: item["name"].lower(),
            )
        except PermissionError:
            raise HTTPException(status_code=403, detail="Folder cannot be read")
        return {
            "path": str(directory),
            "parent": (
                str(directory.parent) if directory != runtime.navigator_root else None
            ),
            "folders": folders,
        }

    def documents_payload(session: ExplorationSession):
        documents, folders = session.documents_and_folders()
        return {
            "documents": documents,
            "folders": folders,
            "documents_default": session.documents_default,
        }

    @app.get("/api/documents")
    async def list_documents(agent_id: str | None = None):
        return documents_payload(active_session(agent_id))

    @app.post("/api/documents", status_code=201)
    async def create_document(
        request: DocumentCreateRequest, agent_id: str | None = None
    ):
        session = active_session(agent_id)
        with http_errors():
            name = session.workspace.create_document(
                request.title, request.text, request.folder
            )
        return {"name": name, **documents_payload(session)}

    @app.post("/api/document-folders", status_code=201)
    async def create_document_folder(
        request: DocumentFolderCreateRequest, agent_id: str | None = None
    ):
        session = active_session(agent_id)
        with http_errors():
            path = session.workspace.create_document_folder(request.path)
        return {"path": path, **documents_payload(session)}

    @app.post("/api/documents/default")
    async def set_documents_default(
        request: DocumentsDefaultRequest, agent_id: str | None = None
    ):
        active_session(agent_id)
        session = runtime.set_documents_default(request.enabled, agent_id)
        return documents_payload(session)

    @app.get("/api/documents/{name:path}")
    async def read_document(name: str, agent_id: str | None = None):
        session = active_session(agent_id)
        with http_errors(not_found="Document not found"):
            return session.workspace.read_document(name)

    @app.put("/api/documents/{name:path}")
    async def write_document(
        name: str, request: DocumentWriteRequest, agent_id: str | None = None
    ):
        session = active_session(agent_id)
        with http_errors(not_found="Document not found"):
            session.workspace.write_document(name, request.text)
            return session.workspace.read_document(name)

    @app.delete("/api/documents/{name:path}")
    async def delete_document(name: str, agent_id: str | None = None):
        active_session(agent_id)
        with http_errors(not_found="Document not found"):
            session = runtime.delete_document(name, agent_id)
        return documents_payload(session)

    @app.post("/api/documents/{name:path}/include")
    async def include_document(
        name: str, request: DocumentIncludeRequest, agent_id: str | None = None
    ):
        session = active_session(agent_id)
        with http_errors(not_found="Document not found"):
            included = session.set_document_included(name, request.included)
        return {"name": name, "included": included}

    @app.get("/api/file")
    async def read_file(path: str, agent_id: str | None = None):
        session = active_session(agent_id)
        try:
            with http_errors(value_status=403, not_found="File not found"):
                return session.read_file(path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/asset")
    async def read_asset(path: str, agent_id: str | None = None):
        session = active_session(agent_id)
        with http_errors(value_status=403, not_found="Asset not found"):
            return FileResponse(session.workspace.asset_file(path))

    @app.get("/api/plots")
    async def list_plots(agent_id: str | None = None):
        session = active_session(agent_id)
        return {
            "plots": session.list_plots(),
            "current_session": session.workspace.session_id,
        }

    @app.get("/api/session/notebook")
    async def download_notebook(agent_id: str | None = None):
        require_idle("download the current session", agent_id)
        with http_errors(value_status=409, not_found_status=409):
            filename, notebook = runtime.export_notebook(agent_id)
        return Response(
            content=json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/kernel/restart")
    async def restart_kernel(agent_id: str | None = None):
        resolved_agent_id, session = require_idle("restart the kernel", agent_id)
        with http_errors():
            alive = await restart_and_broadcast_kernel(resolved_agent_id, session)
        return {"status": "ok", "kernel_alive": alive}

    @app.post("/api/kernel/interrupt")
    async def interrupt_kernel(agent_id: str | None = None):
        session = active_session(agent_id)
        await session.interrupt()
        return {"status": "ok"}

    @app.post("/api/session/defaults")
    async def set_session_defaults(defaults: SessionDefaultsRequest):
        with http_errors():
            runtime.set_session_defaults(
                defaults.model,
                defaults.effort,
                provider=defaults.provider,
                raw_model=defaults.raw_model,
            )
        session = active_session()
        data = {
            "session_defaults": {
                "model": session.default_model,
                "effort": session.default_effort,
            }
        }
        await runtime.broadcast("session_defaults_changed", data)
        return data

    @app.put("/api/settings/exploration-prompt")
    async def set_exploration_prompt(request: ExplorationPromptRequest):
        session = active_session()
        with http_errors():
            prompt = session.set_exploration_prompt(request.prompt)
        data = {
            "exploration_prompt": prompt,
            "exploration_prompt_is_custom": session.exploration_prompt_override
            is not None,
        }
        await runtime.broadcast("exploration_prompt_changed", data)
        return data

    @app.put("/api/settings/prompt-mode")
    async def set_prompt_mode(request: PromptModeRequest):
        active_session()
        with http_errors():
            mode = await runtime.set_prompt_mode(request.mode)
        data = {"prompt_mode": mode}
        await runtime.broadcast("prompt_mode_changed", data)
        return data

    @app.post("/api/session/new")
    async def new_session():
        session = active_session()
        with http_errors():
            await runtime.transition(runtime.reset_session)
        await runtime.broadcast(
            "session_reset",
            {
                "agent_id": runtime.primary_agent_id,
                "session_id": None
                if session.workspace.session_pending
                else session.workspace.session_id,
                "model": session.model,
                "effort": session.effort,
            },
        )
        return {"status": "ok"}

    @app.get("/api/sessions")
    async def list_sessions():
        session = active_session()
        open_agents = {
            candidate.workspace.session_id: agent_id
            for agent_id, candidate in list(runtime.agents.items())
            if not candidate.workspace.session_pending
        }
        sessions = session.list_sessions()
        for item in sessions:
            item["open_agent_id"] = open_agents.get(item["id"])
        return {"sessions": sessions}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        session = active_session()
        with http_errors(not_found="Session not found"):
            return {"id": session_id, "records": session.load_session(session_id)}

    @app.post("/api/session/resume")
    async def resume_session(request: SessionResumeRequest):
        active_session()
        with http_errors(not_found="Session not found"):
            data = await runtime.transition(
                runtime.resume_session, request.id, request.restore_kernel
            )
        payload = {"agent_id": runtime.primary_agent_id, **data}
        await runtime.broadcast("session_resumed", payload)
        return {"status": "ok", **payload}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        if origin and not _same_origin(
            origin, websocket.url.scheme, websocket.headers.get("host", "")
        ):
            await websocket.close(code=1008, reason="WebSocket origin is not allowed.")
            return
        await websocket.accept()
        runtime.websockets.add(websocket)
        try:
            await websocket.send_json(
                {"type": "init", "workspace": runtime.workspace_snapshot()}
            )
        except Exception as exc:
            logger.error(f"Error sending ws init: {exc}")

        try:
            while True:
                data = await websocket.receive_json()
                if not isinstance(data, dict) or app.state.exiting:
                    continue
                action = data.get("action")
                if action != "set_workspace" and runtime.session is None:
                    await ws_error(
                        "Open a workspace folder before using the agent.",
                        data.get("agent_id"),
                    )
                    continue
                if runtime.changing:
                    await ws_error(
                        "The research environment is being prepared. Please wait."
                    )
                    continue

                if action == "new_agent":
                    try:
                        await create_and_broadcast_agent()
                    except Exception as exc:
                        await ws_error(str(exc))
                    continue

                if action == "set_workspace":
                    target_path = data.get("path")
                    if target_path:
                        try:
                            await open_and_broadcast_workspace(
                                target_path, data.get("python")
                            )
                        except Exception as exc:
                            await ws_error(str(exc))
                    continue

                agent_id = data.get("agent_id") or runtime.primary_agent_id
                try:
                    session = runtime.get_agent(agent_id)
                except ValueError as exc:
                    await ws_error(str(exc), agent_id)
                    continue

                if action == "query":
                    raw_text = data.get("text")
                    text = raw_text.strip() if isinstance(raw_text, str) else ""
                    if not text:
                        continue
                    if text == "/new":
                        action = "new_session"
                    elif text == "/restart":
                        action = "restart_kernel"
                    elif text == "/notes":
                        notes = session.workspace.notes.read()
                        await runtime.broadcast(
                            "system_message",
                            {
                                "agent_id": agent_id,
                                "text": f"```markdown\n{notes}\n```",
                            },
                        )
                        continue
                    else:
                        model = data.get("model")
                        provider = data.get("provider")
                        raw_model = data.get("raw_model")
                        if any(
                            value is not None and not isinstance(value, str)
                            for value in (model, provider, raw_model)
                        ):
                            await ws_error(
                                "Choose a model from the model selector.", agent_id
                            )
                            continue
                        effort = data.get("effort")
                        if effort is not None and not isinstance(effort, str):
                            await ws_error(
                                "Choose an effort supported by the selected model.",
                                agent_id,
                            )
                            continue
                        try:
                            query_state = runtime.start_query(
                                text,
                                model,
                                effort,
                                agent_id,
                                provider=provider,
                                raw_model=raw_model,
                            )
                        except (RuntimeError, ValueError) as exc:
                            await ws_error(str(exc), agent_id)
                            continue
                        if query_state == "started":
                            await runtime.broadcast(
                                "user_message",
                                {
                                    "agent_id": agent_id,
                                    "session_id": session.workspace.session_id,
                                    "text": text,
                                    "model": session.model,
                                    "effort": session.effort,
                                },
                            )
                        elif session._last_pending_task is not None:
                            await session._last_pending_task
                        continue

                if action == "cancel_pending_prompt":
                    prompt_id = data.get("prompt_id")
                    if isinstance(prompt_id, str) and prompt_id:
                        await session.remove_pending_prompt(prompt_id)

                elif action == "interrupt":
                    existing = interrupt_tasks.get(agent_id)
                    if existing is not None and not existing.done():
                        task = session._current_task
                        if task is not None and not task.done():
                            task.cancel()
                    else:

                        async def run_interrupt(
                            target_id=agent_id, target_session=session
                        ):
                            try:
                                await target_session.interrupt()
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                logger.exception(
                                    "Interrupt failed for explorer %s", target_id
                                )
                            finally:
                                if (
                                    interrupt_tasks.get(target_id)
                                    is asyncio.current_task()
                                ):
                                    interrupt_tasks.pop(target_id, None)

                        bg = asyncio.create_task(run_interrupt())
                        if not bg.done():
                            interrupt_tasks[agent_id] = bg
                elif action == "restart_kernel":
                    try:
                        await restart_and_broadcast_kernel(agent_id, session)
                    except RuntimeError as exc:
                        await ws_error(str(exc), agent_id)
                elif action == "new_session":
                    try:
                        await create_and_broadcast_agent(requested_by=agent_id)
                    except Exception as exc:
                        await ws_error(str(exc), agent_id)
        except WebSocketDisconnect:
            runtime.websockets.discard(websocket)
        except Exception as exc:
            logger.warning(f"WebSocket error: {exc}")
            runtime.websockets.discard(websocket)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
