"""FastAPI application factory for the Xprober dashboard."""

import asyncio
import logging
import webbrowser
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..session import ExplorationSession
from .runtime import WebRuntime

logger = logging.getLogger("xprober.web")
STATIC_DIR = Path(__file__).parent / "static"


class WorkspaceChangeRequest(BaseModel):
    path: str


class SessionResumeRequest(BaseModel):
    id: str


def create_app(
    session: ExplorationSession | None = None,
    *,
    browser_url: str | None = None,
    model: str = "claude-opus-5",
    timeout_s: int = 600,
    image_cap: int = 2,
    navigator_root: Path | str = ".",
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.start()
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

    app = FastAPI(title="Xprober Web App", lifespan=lifespan)
    app.state.runtime = runtime
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
        session = runtime.session
        if session is None:
            return {
                "is_open": False,
                "path": None,
                "model": runtime.model,
                "kernel_alive": False,
                "kernel_busy": False,
                "is_busy": False,
                "carry_chat_context": True,
                "notes_content": "",
                "markdown_files": [],
                "plots": [],
                "navigator_root": str(runtime.navigator_root),
            }
        notes_content = ""
        if session.notes_path.exists():
            try:
                notes_content = session.notes_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning(f"Could not read {session.notes_path}: {exc}")
        return {
            "is_open": True,
            "path": str(session.workspace_path),
            "model": session.model,
            "kernel_alive": session.kernel.is_alive(),
            "kernel_busy": session.kernel.busy,
            "is_busy": session._is_busy,
            "carry_chat_context": session.carry_chat_context,
            "notes_content": notes_content,
            "markdown_files": session.list_markdown_files(),
            "plots": session.list_plots(),
        }

    def active_session() -> ExplorationSession:
        if runtime.session is None:
            raise HTTPException(
                status_code=409,
                detail="Open a workspace folder before using the agent.",
            )
        return runtime.session

    def require_idle(action: str):
        session = active_session()
        if session.has_active_query():
            raise HTTPException(
                status_code=409,
                detail=f"Cannot {action} while the agent is running a query.",
            )

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_path)

    @app.get("/scratch/{filename}")
    async def get_scratch_plot(filename: str):
        session = active_session()
        file_path = (session.scratch_path / filename).resolve()
        if not str(file_path).startswith(str(session.scratch_path.resolve())):
            raise HTTPException(status_code=403, detail="Forbidden")
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(file_path)

    @app.get("/api/workspace")
    async def get_workspace_info():
        return await workspace_snapshot()

    @app.post("/api/workspace")
    async def change_workspace(request: WorkspaceChangeRequest):
        try:
            runtime.open_workspace(request.path)
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
                    if child.is_dir() and not child.is_symlink()
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

    @app.get("/api/plots")
    async def list_plots():
        session = active_session()
        return {"plots": session.list_plots()}

    @app.post("/api/kernel/restart")
    async def restart_kernel():
        require_idle("restart the kernel")
        session = active_session()
        session.kernel.restart()
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
            session.reset_client_session()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        await runtime.broadcast("session_reset", {})
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
            data = session.resume_session(request.id)
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
                session = runtime.session
                if action != "set_workspace" and session is None:
                    await runtime.broadcast(
                        "error",
                        {"message": "Open a workspace folder before using the agent."},
                    )
                    continue
                if action == "query":
                    text = data.get("text", "").strip()
                    if not text:
                        continue
                    if text == "/new":
                        if session.has_active_query():
                            await runtime.broadcast(
                                "error",
                                {
                                    "message": "Cannot start a new session while the agent is running a query."
                                },
                            )
                            continue
                        session.reset_client_session()
                        await runtime.broadcast(
                            "system_message",
                            {
                                "text": "Started new exploration session with same kernel."
                            },
                        )
                        continue
                    if text == "/restart":
                        if session.has_active_query():
                            await runtime.broadcast(
                                "error",
                                {
                                    "message": "Cannot restart the kernel while the agent is running a query."
                                },
                            )
                            continue
                        session.kernel.restart()
                        await runtime.broadcast(
                            "system_message",
                            {"text": "Kernel restarted. In-memory state is cleared."},
                        )
                        continue
                    if text == "/notes":
                        notes = (
                            session.notes_path.read_text()
                            if session.notes_path.exists()
                            else ""
                        )
                        await runtime.broadcast(
                            "system_message", {"text": f"```markdown\n{notes}\n```"}
                        )
                        continue
                    if session.has_active_query():
                        await runtime.broadcast(
                            "error",
                            {"message": "The agent is already running a query."},
                        )
                        continue
                    await runtime.broadcast("user_message", {"text": text})
                    runtime.start_query(text)

                elif action == "interrupt":
                    await session.interrupt()
                elif action == "restart_kernel":
                    if session.has_active_query():
                        await runtime.broadcast(
                            "error",
                            {
                                "message": "Cannot restart the kernel while the agent is running a query."
                            },
                        )
                        continue
                    session.kernel.restart()
                    await runtime.broadcast(
                        "kernel_restarted", {"kernel_alive": session.kernel.is_alive()}
                    )
                elif action == "new_session":
                    if session.has_active_query():
                        await runtime.broadcast(
                            "error",
                            {
                                "message": "Cannot start a new session while the agent is running a query."
                            },
                        )
                        continue
                    session.reset_client_session()
                    await runtime.broadcast("session_reset", {})
                elif action == "resume_session":
                    session_id = data.get("id")
                    if session_id:
                        try:
                            resumed = session.resume_session(session_id)
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
                            runtime.open_workspace(target_path)
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
