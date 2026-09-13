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

    def active_session() -> ExplorationSession:
        if runtime.session is None:
            raise HTTPException(
                status_code=409,
                detail="Open a workspace folder before using the agent.",
            )
        return runtime.session

    def require_idle(action: str):
        session = active_session()
        if runtime.changing or session.has_active_query():
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
    async def list_documents():
        return documents_payload(active_session())

    @app.post("/api/documents", status_code=201)
    async def create_document(request: DocumentCreateRequest):
        session = active_session()
        try:
            name = session.workspace.create_document(request.title, request.text)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"name": name, **documents_payload(session)}

    @app.post("/api/documents/default")
    async def set_documents_default(request: DocumentsDefaultRequest):
        session = active_session()
        session.set_documents_default(request.enabled)
        return documents_payload(session)

    @app.get("/api/documents/{name}")
    async def read_document(name: str):
        session = active_session()
        try:
            return session.workspace.read_document(name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/documents/{name}")
    async def write_document(name: str, request: DocumentWriteRequest):
        session = active_session()
        try:
            session.workspace.write_document(name, request.text)
            return session.workspace.read_document(name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/documents/{name}")
    async def delete_document(name: str):
        session = active_session()
        try:
            session.delete_document(name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return documents_payload(session)

    @app.post("/api/documents/{name}/include")
    async def include_document(name: str, request: DocumentIncludeRequest):
        session = active_session()
        try:
            included = session.set_document_included(name, request.included)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"name": name, "included": included}

    @app.get("/api/file")
    async def read_file(path: str):
        session = active_session()
        try:
            return session.read_file(path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/asset")
    async def read_asset(path: str):
        session = active_session()
        try:
            return FileResponse(session.workspace.asset_file(path))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Asset not found")
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc))

    @app.get("/api/plots")
    async def list_plots():
        session = active_session()
        return {
            "plots": session.list_plots(),
            "current_session": session.workspace.session_id,
        }

    @app.get("/api/session/notebook")
    async def download_notebook():
        require_idle("download the current session")
        try:
            filename, notebook = runtime.export_notebook()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(
            content=json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/kernel/restart")
    async def restart_kernel():
        require_idle("restart the kernel")
        session = active_session()
        await runtime.transition(session.kernel.restart)
        await runtime.broadcast(
            "kernel_restarted", {"kernel_alive": session.kernel.is_alive()}
        )
        return {"status": "ok", "kernel_alive": session.kernel.is_alive()}

    @app.post("/api/kernel/interrupt")
    async def interrupt_kernel():
        session = active_session()
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
        return {
            "sessions": session.list_sessions(),
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
                session = runtime.session
                if action != "set_workspace" and session is None:
                    await runtime.broadcast(
                        "error",
                        {"message": "Open a workspace folder before using the agent."},
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

                if action == "query":
                    text = data.get("text", "").strip()
                    if not text:
                        continue
                    if text == "/new":
                        if runtime.changing or session.has_active_query():
                            await runtime.broadcast(
                                "error",
                                {
                                    "message": "Cannot start a new session while the agent is running a query."
                                },
                            )
                            continue
                        await runtime.transition(runtime.reset_session)
                        await runtime.broadcast(
                            "session_reset",
                            {
                                "session_id": session.workspace.session_id,
                                "model": session.model,
                                "effort": session.effort,
                            },
                        )
                        continue
                    if text == "/restart":
                        if runtime.changing or session.has_active_query():
                            await runtime.broadcast(
                                "error",
                                {
                                    "message": "Cannot restart the kernel while the agent is running a query."
                                },
                            )
                            continue
                        await runtime.transition(session.kernel.restart)
                        await runtime.broadcast(
                            "system_message",
                            {"text": "Kernel restarted. In-memory state is cleared."},
                        )
                        continue
                    if text == "/notes":
                        notes = session.workspace.notes.read()
                        await runtime.broadcast(
                            "system_message", {"text": f"```markdown\n{notes}\n```"}
                        )
                        continue
                    if runtime.changing or session.has_active_query():
                        await runtime.broadcast(
                            "error",
                            {"message": "The agent is already running a query."},
                        )
                        continue
                    model = data.get("model")
                    if model is not None and model not in runtime.models:
                        await runtime.broadcast(
                            "error",
                            {"message": "Choose a model from the model selector."},
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
                                "message": "Choose an effort supported by the selected model."
                            },
                        )
                        continue
                    await runtime.broadcast("user_message", {"text": text})
                    runtime.start_query(text, model, effort)

                elif action == "interrupt":
                    await session.interrupt()
                elif action == "restart_kernel":
                    if runtime.changing or session.has_active_query():
                        await runtime.broadcast(
                            "error",
                            {
                                "message": "Cannot restart the kernel while the agent is running a query."
                            },
                        )
                        continue
                    await runtime.transition(session.kernel.restart)
                    await runtime.broadcast(
                        "kernel_restarted", {"kernel_alive": session.kernel.is_alive()}
                    )
                elif action == "new_session":
                    if runtime.changing or session.has_active_query():
                        await runtime.broadcast(
                            "error",
                            {
                                "message": "Cannot start a new session while the agent is running a query."
                            },
                        )
                        continue
                    await runtime.transition(runtime.reset_session)
                    await runtime.broadcast(
                        "session_reset",
                        {
                            "session_id": session.workspace.session_id,
                            "model": session.model,
                            "effort": session.effort,
                        },
                    )
                elif action == "resume_session":
                    session_id = data.get("id")
                    if session_id:
                        try:
                            resumed = await runtime.transition(
                                session.resume_session, session_id
                            )
                            await runtime.broadcast("session_resumed", resumed)
                        except Exception as exc:
                            await runtime.broadcast("error", {"message": str(exc)})
                elif action == "set_carry_context":
                    try:
                        enabled = session.set_carry_chat_context(data.get("enabled"))
                    except RuntimeError as exc:
                        await runtime.broadcast("error", {"message": str(exc)})
                        continue
                    await runtime.broadcast(
                        "carry_context_changed", {"carry_chat_context": enabled}
                    )
                elif action == "set_workspace":
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
        except WebSocketDisconnect:
            runtime.websockets.discard(websocket)
        except Exception as exc:
            logger.warning(f"WebSocket error: {exc}")
            runtime.websockets.discard(websocket)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
