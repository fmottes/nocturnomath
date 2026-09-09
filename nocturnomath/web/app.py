"""FastAPI application factory for the Nocturnomath dashboard."""

import asyncio
import json
import logging
import webbrowser
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, SecretStr

from ..session import ExplorationSession
from .runtime import WebRuntime

logger = logging.getLogger("nocturnomath.web")
STATIC_DIR = Path(__file__).parent / "static"


class WorkspaceChangeRequest(BaseModel):
    path: str
    python: str | None = None


class SessionResumeRequest(BaseModel):
    id: str
    restore_kernel: bool = False


class AuthRequest(BaseModel):
    method: Literal["claude_code", "subscription", "api_key"]
    credential: SecretStr | None = None


def create_app(
    session: ExplorationSession | None = None,
    *,
    browser_url: str | None = None,
    model: str = "claude-opus-5",
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
                webbrowser.open(browser_url)

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

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # FastAPI's default 422 body echoes the rejected input, which for
        # /api/auth would be a credential. Keep only the location and message.
        errors = [
            {key: value for key, value in error.items() if key not in ("input", "ctx")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def disable_static_cache(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
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

    def require_same_origin(request: Request, action: str):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(
                status_code=403, detail=f"{action} must be requested from this app."
            )

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
        }

    @app.get("/api/auth")
    async def get_auth():
        return auth_payload()

    @app.post("/api/auth")
    async def set_auth(auth_request: AuthRequest, request: Request):
        require_same_origin(request, "Authentication")
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
    async def exit_service(request: Request, background_tasks: BackgroundTasks):
        require_same_origin(request, "Exit")
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
        try:
            await runtime.transition(
                runtime.open_workspace, request.path, request.python or initial_python
            )
            initial_python = None
            info = await workspace_snapshot()
            await runtime.broadcast("workspace_updated", {"workspace": info})
            return {"status": "ok", "workspace": info}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            logger.exception("Failed to change workspace")
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/folders")
    async def list_folders(path: str | None = None):
        """List child directories for the visual workspace picker without writing."""
        requested = path or str(runtime.navigator_root)
        directory = Path(requested).expanduser().resolve()
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
            "parent": str(directory.parent) if directory.parent != directory else None,
            "folders": folders,
        }

    @app.get("/api/files")
    async def list_files():
        session = active_session()
        return {"files": session.list_markdown_files()}

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

    @app.post("/api/session/new")
    async def new_session():
        try:
            session = active_session()
            await runtime.transition(session.reset_client_session)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        await runtime.broadcast(
            "session_reset", {"session_id": session.workspace.session_id}
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
                        await runtime.transition(session.reset_client_session)
                        await runtime.broadcast(
                            "session_reset",
                            {"session_id": session.workspace.session_id},
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
                    await runtime.broadcast("user_message", {"text": text})
                    runtime.start_query(text, model)

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
                    await runtime.transition(session.reset_client_session)
                    await runtime.broadcast(
                        "session_reset", {"session_id": session.workspace.session_id}
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
