"""
Regression tests for server/asl_model.py -- the shared ML utilities used by
server.py, ser.py, and main.py.

These specifically guard against the two silent, crash-on-start classes of
bug this project hit in practice:

  1. best_model.keras failing to *deserialize* under a newer Keras release
     (ValueError: batch_normalization expects 1 input, received 2) --
     test_model_loads_and_predicts below would fail loudly if that
     regressed, instead of the app just crashing on first request.
  2. mediapipe removing `mp.solutions.hands` from its Python API --
     test_hand_landmarker_initializes below would fail if the Tasks-API
     migration ever broke again.

Run with:  pytest server/tests -v
(from a venv with `pip install -r requirements.txt -r requirements-dev.txt`)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asl_model  # noqa: E402

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(SERVER_DIR, "best_model.keras")
LABELS_PATH = os.path.join(SERVER_DIR, "labels.txt")
HAND_MODEL_PATH = os.path.join(SERVER_DIR, "hand_landmarker.task")

requires_model_assets = pytest.mark.skipif(
    not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)),
    reason="best_model.keras / labels.txt not present in server/",
)
requires_hand_model = pytest.mark.skipif(
    not os.path.exists(HAND_MODEL_PATH),
    reason="hand_landmarker.task not present in server/",
)


# ---------------------------------------------------------------------------
# Pure-logic tests (no model assets required)
# ---------------------------------------------------------------------------

def test_label_for_strips_index_prefix():
    class_names = ["0 A", "1 B", "26 blank", "27 delete", "28 space"]
    assert asl_model.label_for(class_names, 0) == "A"
    assert asl_model.label_for(class_names, 1) == "B"
    assert asl_model.label_for(class_names, 2) == "blank"
    assert asl_model.label_for(class_names, 3) == "delete"
    assert asl_model.label_for(class_names, 4) == "space"


def test_preprocess_image_shape_and_range():
    fake_hand_crop = np.random.randint(0, 256, size=(80, 60, 3), dtype=np.uint8)
    out = asl_model.preprocess_image(fake_hand_crop, target_size=(224, 224))

    assert out.shape == (1, 224, 224, 3)
    assert out.dtype == np.float32
    # MobileNetV2-style preprocessing maps [0, 255] -> [-1, 1]
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_hand_bounding_box_is_padded_and_clamped():
    class Landmark:
        def __init__(self, x, y):
            self.x, self.y = x, y

    # A tight cluster of landmarks near the top-left corner of a 100x100 frame.
    landmarks = [Landmark(0.05, 0.05), Landmark(0.10, 0.10)]
    x_min, y_min, x_max, y_max = asl_model.hand_bounding_box(
        landmarks, frame_shape=(100, 100, 3), padding=20
    )

    # Padding pushes the box past the frame edge; it must clamp to 0, not go negative.
    assert x_min == 0
    assert y_min == 0
    assert x_max > x_min
    assert y_max > y_min
    assert x_max <= 100
    assert y_max <= 100


# ---------------------------------------------------------------------------
# Model-asset tests (skipped automatically if the committed model files are
# unavailable, e.g. in a minimal checkout, but run by default in this repo
# and in CI since best_model.keras/labels.txt/hand_landmarker.task are
# committed).
# ---------------------------------------------------------------------------

@requires_model_assets
def test_labels_file_has_29_classes():
    class_names = asl_model.load_labels(LABELS_PATH)
    assert len(class_names) == 29
    assert asl_model.label_for(class_names, 26) == "blank"
    assert asl_model.label_for(class_names, 27) == "delete"
    assert asl_model.label_for(class_names, 28) == "space"


@requires_model_assets
def test_model_loads_and_predicts():
    model = asl_model.load_model(MODEL_PATH)
    class_names = asl_model.load_labels(LABELS_PATH)

    fake_hand_crop = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    input_data = asl_model.preprocess_image(fake_hand_crop)
    prediction = model.predict(input_data, verbose=0)

    # One softmax row over all 29 classes that (approximately) sums to 1.
    assert prediction.shape == (1, len(class_names))
    assert prediction.min() >= 0.0
    assert np.isclose(prediction.sum(), 1.0, atol=1e-3)

    index = int(np.argmax(prediction))
    label = asl_model.label_for(class_names, index)
    assert label in class_names[index]


@requires_hand_model
def test_hand_landmarker_initializes():
    """Guards against the mediapipe.solutions.hands removal regressing:
    if the Tasks API is ever broken again, this fails immediately instead
    of every backend silently crashing on startup."""
    landmarker = asl_model.create_hand_landmarker(HAND_MODEL_PATH, num_hands=2)
    assert landmarker is not None

    # An all-black frame has no hand in it; detection should run without
    # raising and simply report no hands found.
    blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    hands = asl_model.detect_hands(landmarker, blank_frame)
    assert hands == []
