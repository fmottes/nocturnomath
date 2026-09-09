"""Persistent Jupyter kernel used by the exploration tools."""

import base64
import queue
import re
import time
from pathlib import Path

from jupyter_client import KernelManager

# All CSI escapes, not just colours: uv's installer emits cursor moves and line erases too.
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

SETUP = """
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.dpi"] = 110
"""


class Kernel:
    def __init__(self, cwd=None):
        self.km = None
        self.kc = None
        self.busy = False  # read by interrupt handlers in both interfaces
        self.errored = False  # whether the last execute() raised; see execute()
        self.cwd = Path(cwd).resolve() if cwd else None
        self.start()

    def start(self):
        self.km = KernelManager()
        start_kwargs = {"cwd": str(self.cwd)} if self.cwd else {}
        self.km.start_kernel(**start_kwargs)
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=60)
        setup_code = SETUP
        if self.cwd:
            setup_code += f"\nimport os\nos.chdir({str(self.cwd)!r})\n"
        self.execute(setup_code, timeout=60)

    def set_cwd(self, cwd):
        """Change the current working directory of the running kernel."""
        self.cwd = Path(cwd).resolve()
        if self.is_alive():
            self.execute(f"import os\nos.chdir({str(self.cwd)!r})", timeout=10)

    def restart(self):
        self.shutdown()
        self.start()

    def shutdown(self):
        if self.kc is not None:
            self.kc.stop_channels()
        if self.km is not None:
            self.km.shutdown_kernel(now=True)

    def is_alive(self):
        return self.km is not None and self.km.is_alive()

    def interrupt(self):
        if self.km is not None:
            self.km.interrupt_kernel()

    def execute(self, code, timeout):
        """Run code. Returns (text output, [png bytes], note or None).

        Whether the cell raised is left on `self.errored` rather than returned, so
        callers that do not care keep the three-value signature.
        """
        self.busy = True
        self.errored = False
        try:
            return self._execute(code, timeout)
        finally:
            self.busy = False

    def _execute(self, code, timeout):
        msg_id = self.kc.execute(code)
        parts = []
        images = []
        note = None
        deadline = time.time() + timeout

        while True:
            left = deadline - time.time()
            if left <= 0:
                if note is not None:  # the grace period after an interrupt ran out too
                    if self.is_alive():
                        self.restart()
                        note += " Kernel did not become idle and was restarted; in-memory state is gone."
                    break
                self.interrupt()
                note = f"[timed out after {timeout}s; kernel interrupted, partial output above]"
                deadline = time.time() + 10  # let the interrupt traceback arrive
                continue

            try:
                msg = self.kc.get_iopub_msg(timeout=min(0.5, left))
            except queue.Empty:
                if not self.is_alive():
                    note = "[the kernel died during this run; in-memory state is gone]"
                    break
                continue

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            kind = msg["msg_type"]
            content = msg["content"]
            if kind == "stream":
                parts.append(content["text"])
            elif kind in ("execute_result", "display_data"):
                data = content["data"]
                if "image/png" in data:
                    images.append(base64.b64decode(data["image/png"]))
                elif "text/plain" in data:
                    parts.append(data["text/plain"] + "\n")
            elif kind == "error":
                self.errored = True
                parts.append("\n".join(content["traceback"]) + "\n")
            elif kind == "status" and content["execution_state"] == "idle":
                break

        return ANSI.sub("", "".join(parts)), images, note
