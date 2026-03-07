"""RT-Gesture constants and enums."""

from __future__ import annotations

import enum

EMG_NUM_CHANNELS = 16
EMG_SAMPLE_RATE = 2000  # Hz
EMG_DTYPE = "float32"

MODEL_STRIDE = 10
MODEL_LEFT_CONTEXT = 20
MODEL_OUTPUT_RATE = EMG_SAMPLE_RATE / MODEL_STRIDE
MODEL_OUTPUT_CHANNELS = 9


class GestureType(enum.Enum):
    """Discrete gesture classes and output indices."""

    index_press = 0
    index_release = 1
    middle_press = 2
    middle_release = 3
    thumb_click = 4
    thumb_down = 5
    thumb_in = 6
    thumb_out = 7
    thumb_up = 8


GESTURE_NAMES = [gesture.name for gesture in GestureType]

DEFAULT_THRESHOLD = 0.35
DEFAULT_REJECTION_THRESHOLD = 0.6
DEFAULT_REST_THRESHOLD = 0.15
DEFAULT_DEBOUNCE_SEC = 0.05

ZMQ_EMG_PORT = 5555
ZMQ_GT_PORT = 5556
ZMQ_RESULT_PORT = 5557
ZMQ_CONTROL_PORT = 5558


class MsgType:
    """ZMQ message type names."""

    EMG_CHUNK = "emg_chunk"
    GROUND_TRUTH = "ground_truth"
    PROBABILITIES = "probabilities"
    GESTURE_EVENT = "gesture_event"
    SHUTDOWN = "shutdown"
    HEARTBEAT = "heartbeat"


DATA_TIMEOUT_MS = 500
WARM_UP_FRAMES = 20
MAX_CONSECUTIVE_DROPS = 3
SHUTDOWN_TIMEOUT_SEC = 5.0

DEFAULT_HEARTBEAT_INTERVAL_SEC = 1.0
DEFAULT_HEARTBEAT_TIMEOUT_SEC = 3.0

LATENCY_WARN_TRANSPORT_MS = 5.0
LATENCY_WARN_INFER_MS = 10.0
LATENCY_WARN_POST_MS = 5.0
LATENCY_WARN_PIPELINE_MS = 80.0
