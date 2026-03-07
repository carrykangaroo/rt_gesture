"""Backend process management for GUI controls."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import zmq

from rt_gesture.constants import MsgType
from rt_gesture.zmq_transport import ZmqPublisher


class ProcessManager:
    """Manage lifecycle of the backend realtime pipeline process."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._workspace_dir = Path(__file__).resolve().parents[2]

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, config_path: str | Path) -> bool:
        """Start backend pipeline via `python -m rt_gesture.main`."""
        command = [sys.executable, "-m", "rt_gesture.main", str(config_path)]
        return self.start_command(command, cwd=self._workspace_dir)

    def start_command(self, command: list[str], cwd: str | Path | None = None) -> bool:
        """Start a process using an explicit command, useful for tests."""
        if self.is_running():
            return False
        run_cwd = Path(cwd) if cwd is not None else self._workspace_dir
        self._process = subprocess.Popen(command, cwd=str(run_cwd))
        return True

    def request_shutdown(self, control_port: int) -> None:
        """Send SHUTDOWN control message to backend subscribers."""
        ctx = zmq.Context()
        publisher = ZmqPublisher(ctx, control_port, bind=True)
        time.sleep(0.1)
        publisher.send(MsgType.SHUTDOWN)
        publisher.close()
        ctx.term()

    def stop(self, control_port: int, timeout_sec: float) -> bool:
        """Request graceful shutdown then enforce terminate/kill if needed."""
        if not self.is_running():
            return False

        self.request_shutdown(control_port)
        assert self._process is not None
        try:
            self._process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        finally:
            self._process = None
        return True

