from rt_gesture.constants import (
    EMG_NUM_CHANNELS,
    EMG_SAMPLE_RATE,
    GESTURE_NAMES,
    GestureType,
    MODEL_OUTPUT_CHANNELS,
    MODEL_OUTPUT_RATE,
    MODEL_STRIDE,
    ZMQ_CONTROL_PORT,
    ZMQ_EMG_PORT,
    ZMQ_GT_PORT,
    ZMQ_RESULT_PORT,
)


def test_gesture_enum_matches_expected_order() -> None:
    expected_names = [
        "index_press",
        "index_release",
        "middle_press",
        "middle_release",
        "thumb_click",
        "thumb_down",
        "thumb_in",
        "thumb_out",
        "thumb_up",
    ]
    assert [gesture.name for gesture in GestureType] == expected_names
    assert [gesture.value for gesture in GestureType] == list(range(9))
    assert GESTURE_NAMES == expected_names


def test_signal_and_model_constants() -> None:
    assert EMG_NUM_CHANNELS == 16
    assert EMG_SAMPLE_RATE == 2000
    assert MODEL_STRIDE == 10
    assert MODEL_OUTPUT_CHANNELS == 9
    assert MODEL_OUTPUT_RATE == 200


def test_default_ports_are_unique() -> None:
    ports = {ZMQ_EMG_PORT, ZMQ_GT_PORT, ZMQ_RESULT_PORT, ZMQ_CONTROL_PORT}
    assert len(ports) == 4

