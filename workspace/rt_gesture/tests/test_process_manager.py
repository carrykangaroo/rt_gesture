from __future__ import annotations

import socket
import sys
import time

from rt_gesture.gui.process_manager import ProcessManager


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_start_command_prevents_duplicate_start() -> None:
    manager = ProcessManager()
    started = manager.start_command([sys.executable, "-c", "import time; time.sleep(30)"])
    assert started
    try:
        assert manager.is_running()
        started_again = manager.start_command([sys.executable, "-c", "print('x')"])
        assert not started_again
    finally:
        manager.stop(control_port=_pick_free_port(), timeout_sec=0.1)


def test_stop_falls_back_to_terminate_when_no_shutdown_subscriber() -> None:
    manager = ProcessManager()
    started = manager.start_command([sys.executable, "-c", "import time; time.sleep(30)"])
    assert started
    assert manager.is_running()

    stopped = manager.stop(control_port=_pick_free_port(), timeout_sec=0.1)
    assert stopped
    time.sleep(0.1)
    assert not manager.is_running()

