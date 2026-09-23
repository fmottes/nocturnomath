"""FastAPI application factory for the Nocturnomath dashboard."""

import asyncio
import json
import logging
import webbrowser
from collections.abc import Callable
from contextlib import asynccontextmanager
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


class DocumentWriteRequest(BaseModel):
    text: str


class DocumentIncludeRequest(BaseModel):
    included: bool


class DocumentsDefaultRequest(BaseModel):
    enabled: bool


class SessionDefaultsRequest(BaseModel):
    model: str
    effort: str | None = None


class AuthRequest(BaseModel):
    method: Literal["claude_code", "subscription", "api_key"]
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
    )

    initial_python = python

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.start()
        await runtime.discover_models()
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

    async def workspace_snapshot():
        return runtime.workspace_snapshot()

    def active_session(agent_id: str | None = None) -> ExplorationSession:
        if runtime.session is None:
            raise HTTPException(
                status_code=409,
                detail="Open a workspace folder before using the agent.",
            )
        try:
            return runtime.get_agent(agent_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def require_idle(action: str, agent_id: str | None = None):
        session = active_session(agent_id)
        if (
            runtime.changing
            or agent_id in runtime._changing_agents
            or session.has_active_query()
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot {action} while the agent is running a query.",
            )

    def navigation_path(path: Path | str) -> Path:
        target = Path(path).expanduser().resolve()
        if not target.is_relative_to(runtime.navigator_root):
            raise PermissionError("Folder is outside the configured workspace root.")
        return target

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_path)

    @app.get("/api/workspace")
    async def get_workspace_info():
        return await workspace_snapshot()

    @app.get("/api/agents/{agent_id}")
    async def get_agent(agent_id: str):
        try:
            return runtime.agent_snapshot(agent_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/agents", status_code=201)
    async def create_agent():
        if runtime.session is None:
            raise HTTPException(
                status_code=409,
                detail="Open a workspace folder before using the agent.",
            )
        try:
            agent_id = await asyncio.to_thread(runtime.create_agent)
            snapshot = runtime.agent_snapshot(agent_id)
        except Exception as exc:
            logger.exception("Failed to create explorer")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await runtime.broadcast("agent_added", snapshot)
        return snapshot

    @app.delete("/api/agents/{agent_id}")
    async def close_agent(agent_id: str):
        try:
            await runtime.transition_agent(agent_id, runtime.close_agent, agent_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        await runtime.broadcast("agent_removed", {"agent_id": agent_id})
        return {"status": "ok"}

    @app.post("/api/agents/{agent_id}/resume")
    async def replace_agent(agent_id: str, request: SessionResumeRequest):
        try:
            resolved_id, data, focused = await runtime.transition_agent(
                agent_id,
                runtime.replace_agent,
                agent_id,
                request.id,
                request.restore_kernel,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        event = "agent_focused" if focused else "session_resumed"
        payload = {"agent_id": resolved_id, **data}
        await runtime.broadcast(event, payload)
        return {"status": "ok", "focused_existing": focused, **payload}

    @app.post("/api/agents/{agent_id}/compact")
    async def compact_agent(agent_id: str):
        try:
            await runtime.compact_agent(agent_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
            "auth": runtime.auth.public(),
            "model": runtime.session.model if runtime.session else runtime.model,
            "models": runtime.models,
            "model_labels": runtime.model_labels,
            "model_efforts": runtime.model_efforts,
            "effort": runtime.session.effort if runtime.session else runtime.effort,
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
        try:
            await runtime.authenticate(auth_request.method, credential)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        data = auth_payload()
        await runtime.broadcast("auth_changed", data)
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
        nonlocal initial_python
        if runtime.session is not None:
            require_idle("change workspace")
        try:
            target = navigation_path(request.path)
            await runtime.transition(
                runtime.open_workspace, target, request.python or initial_python
            )
            initial_python = None
            info = await workspace_snapshot()
            await runtime.broadcast("workspace_updated", {"workspace": info})
            return {"status": "ok", "workspace": info}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:
            logger.exception("Failed to change workspace")
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/folders")
    async def list_folders(path: str | None = None):
        """List child directories for the visual workspace picker without writing."""
        requested = path or str(runtime.navigator_root)
        try:
            directory = navigation_path(requested)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
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
        return {
            "documents": session.list_documents(),
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
        try:
            name = session.workspace.create_document(request.title, request.text)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"name": name, **documents_payload(session)}

    @app.post("/api/documents/default")
    async def set_documents_default(
        request: DocumentsDefaultRequest, agent_id: str | None = None
    ):
        session = active_session(agent_id)
        session.set_documents_default(request.enabled)
        for other in runtime.agents.values():
            if other is not session:
                other.documents_default = session.documents_default
                other.document_choices = {}
        return documents_payload(session)

    @app.get("/api/documents/{name}")
    async def read_document(name: str, agent_id: str | None = None):
        session = active_session(agent_id)
        try:
            return session.workspace.read_document(name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/documents/{name}")
    async def write_document(
        name: str, request: DocumentWriteRequest, agent_id: str | None = None
    ):
        session = active_session(agent_id)
        try:
            session.workspace.write_document(name, request.text)
            return session.workspace.read_document(name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/documents/{name}")
    async def delete_document(name: str, agent_id: str | None = None):
        session = active_session(agent_id)
        try:
            session.delete_document(name)
            for other in runtime.agents.values():
                if other is not session:
                    other.document_choices.pop(name, None)
                    other._sent_documents.pop(name, None)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return documents_payload(session)

    @app.post("/api/documents/{name}/include")
    async def include_document(
        name: str, request: DocumentIncludeRequest, agent_id: str | None = None
    ):
        session = active_session(agent_id)
        try:
            included = session.set_document_included(name, request.included)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"name": name, "included": included}

    @app.get("/api/file")
    async def read_file(path: str, agent_id: str | None = None):
        session = active_session(agent_id)
        try:
            return session.read_file(path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/asset")
    async def read_asset(path: str, agent_id: str | None = None):
        session = active_session(agent_id)
        try:
            return FileResponse(session.workspace.asset_file(path))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Asset not found")
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc))

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
        try:
            filename, notebook = runtime.export_notebook(agent_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(
            content=json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/kernel/restart")
    async def restart_kernel(agent_id: str | None = None):
        require_idle("restart the kernel", agent_id)
        session = active_session(agent_id)
        await runtime.transition_agent(
            agent_id or runtime.primary_agent_id, session.kernel.restart
        )
        await runtime.broadcast(
            "kernel_restarted",
            {"agent_id": agent_id, "kernel_alive": session.kernel.is_alive()},
        )
        return {"status": "ok", "kernel_alive": session.kernel.is_alive()}

    @app.post("/api/kernel/interrupt")
    async def interrupt_kernel(agent_id: str | None = None):
        session = active_session(agent_id)
        await session.interrupt()
        return {"status": "ok"}

    @app.post("/api/session/defaults")
    async def set_session_defaults(defaults: SessionDefaultsRequest):
        try:
            runtime.set_session_defaults(defaults.model, defaults.effort)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        session = active_session()
        data = {
            "session_defaults": {
                "model": session.default_model,
                "effort": session.default_effort,
            }
        }
        await runtime.broadcast("session_defaults_changed", data)
        return data

    @app.post("/api/session/new")
    async def new_session():
        try:
            session = active_session()
            await runtime.transition(runtime.reset_session)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        await runtime.broadcast(
            "session_reset",
            {
                "session_id": session.workspace.session_id,
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
            for agent_id, candidate in runtime.agents.items()
            if not candidate.workspace.session_pending
        }
        sessions = session.list_sessions()
        for item in sessions:
            item["open_agent_id"] = open_agents.get(item["id"])
        return {
            "sessions": sessions,
            "carry_chat_context": session.carry_chat_context,
        }

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        session = active_session()
        try:
            return {"id": session_id, "records": session.load_session(session_id)}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/session/resume")
    async def resume_session(request: SessionResumeRequest):
        try:
            session = active_session()
            data = await runtime.transition(
                session.resume_session, request.id, request.restore_kernel
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        await runtime.broadcast("session_resumed", data)
        return {"status": "ok", **data}

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
            info = await workspace_snapshot()
            await websocket.send_json({"type": "init", "workspace": info})
        except Exception as exc:
            logger.error(f"Error sending ws init: {exc}")

        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action")
                if app.state.exiting:
                    continue
                if action != "set_workspace" and runtime.session is None:
                    error = {
                        "message": "Open a workspace folder before using the agent."
                    }
                    if data.get("agent_id") is not None:
                        error["agent_id"] = data["agent_id"]
                    await runtime.broadcast(
                        "error",
                        error,
                    )
                    continue
                if runtime.changing:
                    await runtime.broadcast(
                        "error",
                        {
                            "message": "The research environment is being prepared. Please wait."
                        },
                    )
                    continue

                if action == "new_agent":
                    try:
                        agent_id = await asyncio.to_thread(runtime.create_agent)
                        await runtime.broadcast(
                            "agent_added", runtime.agent_snapshot(agent_id)
                        )
                    except Exception as exc:
                        await runtime.broadcast("error", {"message": str(exc)})
                    continue

                if action == "set_workspace":
                    target_path = data.get("path")
                    if target_path:
                        try:
                            target_path = navigation_path(target_path)
                            await runtime.transition(
                                runtime.open_workspace, target_path, data.get("python")
                            )
                            info = await workspace_snapshot()
                            await runtime.broadcast(
                                "workspace_updated", {"workspace": info}
                            )
                        except Exception as exc:
                            await runtime.broadcast("error", {"message": str(exc)})
                    continue

                agent_id = data.get("agent_id") or runtime.primary_agent_id
                try:
                    session = runtime.get_agent(agent_id)
                except ValueError as exc:
                    await runtime.broadcast(
                        "error", {"agent_id": agent_id, "message": str(exc)}
                    )
                    continue

                if action == "query":
                    text = data.get("text", "").strip()
                    if not text:
                        continue
                    if text == "/new":
                        try:
                            new_id = await asyncio.to_thread(runtime.create_agent)
                            await runtime.broadcast(
                                "agent_added",
                                {
                                    "requested_by": agent_id,
                                    **runtime.agent_snapshot(new_id),
                                },
                            )
                        except Exception as exc:
                            await runtime.broadcast(
                                "error", {"agent_id": agent_id, "message": str(exc)}
                            )
                        continue
                    if text == "/restart":
                        try:
                            await runtime.transition_agent(
                                agent_id, session.kernel.restart
                            )
                            await runtime.broadcast(
                                "system_message",
                                {
                                    "agent_id": agent_id,
                                    "text": "Kernel restarted. In-memory state is cleared.",
                                },
                            )
                        except RuntimeError as exc:
                            await runtime.broadcast(
                                "error", {"agent_id": agent_id, "message": str(exc)}
                            )
                        continue
                    if text == "/notes":
                        notes = session.workspace.notes.read()
                        await runtime.broadcast(
                            "system_message",
                            {
                                "agent_id": agent_id,
                                "text": f"```markdown\n{notes}\n```",
                            },
                        )
                        continue
                    model = data.get("model")
                    if model is not None and model not in runtime.models:
                        await runtime.broadcast(
                            "error",
                            {
                                "agent_id": agent_id,
                                "message": "Choose a model from the model selector.",
                            },
                        )
                        continue
                    effort = data.get("effort")
                    target_model = model or session.model
                    if effort is not None and effort not in runtime.model_efforts.get(
                        target_model or "", []
                    ):
                        await runtime.broadcast(
                            "error",
                            {
                                "agent_id": agent_id,
                                "message": "Choose an effort supported by the selected model.",
                            },
                        )
                        continue
                    try:
                        runtime.start_query(text, model, effort, agent_id)
                    except (RuntimeError, ValueError) as exc:
                        await runtime.broadcast(
                            "error", {"agent_id": agent_id, "message": str(exc)}
                        )
                        continue
                    await runtime.broadcast(
                        "user_message",
                        {
                            "agent_id": agent_id,
                            "session_id": session.workspace.session_id,
                            "text": text,
                        },
                    )

                elif action == "interrupt":
                    await session.interrupt()
                elif action == "restart_kernel":
                    try:
                        await runtime.transition_agent(agent_id, session.kernel.restart)
                        await runtime.broadcast(
                            "kernel_restarted",
                            {
                                "agent_id": agent_id,
                                "kernel_alive": session.kernel.is_alive(),
                            },
                        )
                    except RuntimeError as exc:
                        await runtime.broadcast(
                            "error", {"agent_id": agent_id, "message": str(exc)}
                        )
                elif action == "new_session":
                    new_id = await asyncio.to_thread(runtime.create_agent)
                    await runtime.broadcast(
                        "agent_added",
                        {"requested_by": agent_id, **runtime.agent_snapshot(new_id)},
                    )
                elif action == "set_carry_context":
                    try:
                        enabled = session.set_carry_chat_context(data.get("enabled"))
                    except RuntimeError as exc:
                        await runtime.broadcast(
                            "error", {"agent_id": agent_id, "message": str(exc)}
                        )
                        continue
                    await runtime.broadcast(
                        "carry_context_changed",
                        {"agent_id": agent_id, "carry_chat_context": enabled},
                    )
        except WebSocketDisconnect:
            runtime.websockets.discard(websocket)
        except Exception as exc:
            logger.warning(f"WebSocket error: {exc}")
            runtime.websockets.discard(websocket)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
