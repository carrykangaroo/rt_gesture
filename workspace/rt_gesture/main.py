"""Process orchestration entrypoint for RT-Gesture."""

from __future__ import annotations

import logging
import multiprocessing as mp
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


def run_data_simulator(config: AppConfig) -> None:
    _configure_worker_logging(config.logging.log_level)
    simulator = DataSimulator(config.data_simulator)
    simulator.load_data(config.data_simulator.hdf5_path)
    simulator.run(max_chunks=config.data_simulator.max_chunks)


def run_inference_engine(config: AppConfig, run_dir: str) -> None:
    _configure_worker_logging(config.logging.log_level)
    engine = InferenceEngine(config.inference, run_dir=run_dir)
    engine.run(max_messages=config.inference.max_messages)


def _send_shutdown(control_port: int) -> None:
    ctx = zmq.Context()
    pub = ZmqPublisher(ctx, control_port, bind=True)
    time.sleep(0.1)
    pub.send(MsgType.SHUTDOWN)
    pub.close()
    ctx.term()


def main(config_path: str = "config/default.yaml") -> None:
    config = load_config(config_path)
    run_dir = setup_logging(config.logging.log_dir, config.logging.log_level)
    log.info("RT-Gesture run directory: %s", run_dir)

    inference_proc = mp.Process(
        target=run_inference_engine,
        args=(config, str(run_dir)),
        name="InferenceEngine",
        daemon=False,
    )
    inference_proc.start()
    log.info("InferenceEngine started (pid=%s)", inference_proc.pid)

    time.sleep(0.5)
    data_proc = mp.Process(
        target=run_data_simulator,
        args=(config,),
        name="DataSimulator",
        daemon=False,
    )
    data_proc.start()
    log.info("DataSimulator started (pid=%s)", data_proc.pid)

    try:
        data_proc.join()
        log.info("DataSimulator exited")
        _send_shutdown(config.inference.control_port)
        inference_proc.join(timeout=config.shutdown_timeout_sec)
        if inference_proc.is_alive():
            log.warning("InferenceEngine timeout, terminating")
            inference_proc.terminate()
            inference_proc.join(timeout=2)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received, shutting down")
        _send_shutdown(config.inference.control_port)
        data_proc.join(timeout=config.shutdown_timeout_sec)
        inference_proc.join(timeout=config.shutdown_timeout_sec)
        for proc in (data_proc, inference_proc):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

    log.info("RT-Gesture shutdown complete")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml")
