"""Process orchestration entrypoint for RT-Gesture."""

from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import socket
import sys
import time
from pathlib import Path

import zmq

from rt_gesture.config import AppConfig, load_config
from rt_gesture.constants import MsgType
from rt_gesture.data_simulator import DataSimulator
from rt_gesture.inference_engine import InferenceEngine
from rt_gesture.logger import setup_logging
from rt_gesture.zmq_transport import ZmqPublisher

log = logging.getLogger(__name__)


def _configure_worker_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _set_parent_death_signal() -> None:
    """On Linux, ask kernel to send SIGTERM when parent process dies."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
            errno = ctypes.get_errno()
            log.debug("prctl(PR_SET_PDEATHSIG) failed errno=%s", errno)
    except Exception:
        log.debug("Failed to configure parent-death signal", exc_info=True)


def run_data_simulator(config: AppConfig) -> None:
    _set_parent_death_signal()
    _configure_worker_logging(config.logging.log_level)
    simulator = DataSimulator(config.data_simulator)
    simulator.load_data(config.data_simulator.hdf5_path)
    simulator.run(max_chunks=config.data_simulator.max_chunks)


def run_inference_engine(config: AppConfig, run_dir: str) -> None:
    _set_parent_death_signal()
    _configure_worker_logging(config.logging.log_level)
    engine = InferenceEngine(config.inference, run_dir=run_dir)
    engine.run(max_messages=config.inference.max_messages)


def _wait_port_available(port: int, timeout_sec: float = 5.0, poll_interval_sec: float = 0.1) -> bool:
    """Return True once the TCP port can be bound locally within timeout."""
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    while time.monotonic() <= deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(poll_interval_sec)
    return False


def _runtime_ports(config: AppConfig) -> set[int]:
    return {
        config.data_simulator.emg_port,
        config.data_simulator.gt_port,
        config.data_simulator.control_port,
        config.inference.emg_port,
        config.inference.result_port,
        config.inference.control_port,
    }


def _ensure_runtime_ports_available(config: AppConfig, timeout_sec: float = 5.0) -> None:
    for port in sorted(_runtime_ports(config)):
        if _wait_port_available(port, timeout_sec=timeout_sec):
            continue
        raise RuntimeError(
            f"Port {port} is not available after waiting {timeout_sec:.1f}s; "
            "a previous process may still be holding it."
        )


def _send_shutdown(control_port: int, timeout_sec: float = 5.0) -> bool:
    if not _wait_port_available(control_port, timeout_sec=timeout_sec):
        log.warning("Control port %s is unavailable; cannot publish SHUTDOWN", control_port)
        return False

    ctx = zmq.Context()
    publisher: ZmqPublisher | None = None
    try:
        publisher = ZmqPublisher(ctx, control_port, bind=True)
        # PUB sockets need a brief window for subscribers to receive first message.
        time.sleep(0.1)
        publisher.send(MsgType.SHUTDOWN)
        return True
    except zmq.ZMQError:
        log.exception("Failed to publish SHUTDOWN on control port %s", control_port)
        return False
    finally:
        if publisher is not None:
            publisher.close()
        ctx.term()


def _stop_process(proc: mp.Process, timeout_sec: float, name: str) -> None:
    if not proc.is_alive():
        return
    proc.join(timeout=timeout_sec)
    if proc.is_alive():
        log.warning("%s did not exit in %.1fs, terminating", name, timeout_sec)
        proc.terminate()
        proc.join(timeout=2.0)
    if proc.is_alive():
        log.warning("%s still alive after terminate, killing", name)
        proc.kill()
        proc.join(timeout=2.0)


def main(config_path: str = "config/default.yaml") -> None:
    config = load_config(config_path)
    _ensure_runtime_ports_available(config, timeout_sec=config.shutdown_timeout_sec)

    run_dir = setup_logging(config.logging.log_dir, config.logging.log_level)
    log.info("RT-Gesture run directory: %s", run_dir)

    shutdown_requested = False

    def _signal_handler(signum: int, _frame: object) -> None:
        nonlocal shutdown_requested
        if not shutdown_requested:
            log.info("Received signal %s, initiating graceful shutdown", signum)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    inference_proc = mp.Process(
        target=run_inference_engine,
        args=(config, str(run_dir)),
        name="InferenceEngine",
        daemon=False,
    )
    data_proc: mp.Process | None = None
    should_send_shutdown = False

    try:
        inference_proc.start()
        log.info("InferenceEngine started (pid=%s)", inference_proc.pid)

        if shutdown_requested:
            log.info("Shutdown requested before DataSimulator start")
        else:
            time.sleep(0.5)
            data_proc = mp.Process(
                target=run_data_simulator,
                args=(config,),
                name="DataSimulator",
                daemon=False,
            )
            data_proc.start()
            log.info("DataSimulator started (pid=%s)", data_proc.pid)

            while data_proc.is_alive() and not shutdown_requested:
                data_proc.join(timeout=0.2)
            if data_proc.is_alive():
                log.info("Shutdown requested before DataSimulator completed")
            else:
                log.info("DataSimulator exited (code=%s)", data_proc.exitcode)
        should_send_shutdown = True
    except KeyboardInterrupt:
        shutdown_requested = True
        should_send_shutdown = True
        log.info("KeyboardInterrupt received, shutting down")
    finally:
        if should_send_shutdown and inference_proc.is_alive():
            _send_shutdown(config.inference.control_port, timeout_sec=config.shutdown_timeout_sec)

        if data_proc is not None:
            _stop_process(data_proc, timeout_sec=config.shutdown_timeout_sec, name="DataSimulator")
        _stop_process(inference_proc, timeout_sec=config.shutdown_timeout_sec, name="InferenceEngine")

    log.info("RT-Gesture shutdown complete")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml")
