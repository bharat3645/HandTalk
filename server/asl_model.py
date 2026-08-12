"""
Shared ASL hand-detection + classification utilities.

server.py, ser.py, and main.py each used to carry their own hand-copied
version of "load the Keras model", "load labels.txt", "crop the hand region
and preprocess it", and "run MediaPipe HandLandmarker" -- three copies of the
same logic that could (and did, historically) drift out of sync. This module
is the single implementation all three backends import, so:

  - fixing a bug (e.g. the Keras/MediaPipe API-compat issues) fixes it
    everywhere at once instead of requiring three coordinated edits, and
  - the core ML logic can be unit-tested (see tests/test_asl_model.py)
    without having to boot a Flask/FastAPI app or open a webcam.
"""

import os
import logging

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "./best_model.keras")
DEFAULT_LABELS_PATH = os.getenv("LABELS_PATH", "./labels.txt")
DEFAULT_HAND_MODEL_PATH = os.getenv("HAND_MODEL_PATH", "./hand_landmarker.task")

# Special, non-alphabetic classes the model was trained on (see labels.txt).
BLANK_LABEL = "blank"
DELETE_LABEL = "delete"
SPACE_LABEL = "space"


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load the trained Keras ASL classifier."""
    logger.info("Loading model from %s...", model_path)
    return tf.keras.models.load_model(model_path)


def load_labels(labels_path: str = DEFAULT_LABELS_PATH):
    """Load class labels. labels.txt stores 'index label' pairs, e.g. '5 F'."""
    logger.info("Loading labels from %s...", labels_path)
    with open(labels_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def label_for(class_names, index: int) -> str:
    """Strip the leading index off a labels.txt entry, e.g. '5 F' -> 'F'."""
    return class_names[index].split(" ", 1)[-1]


def create_hand_landmarker(
    hand_model_path: str = DEFAULT_HAND_MODEL_PATH,
    num_hands: int = 2,
    min_hand_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Create a MediaPipe Tasks HandLandmarker.

    mediapipe removed the legacy `mp.solutions.hands` API from recent
    releases; hand detection now goes through the Tasks API instead.
    """
    logger.info("Initializing MediaPipe HandLandmarker from %s...", hand_model_path)
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path),
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
    )


def detect_hands(hand_landmarker, rgb_image: np.ndarray):
    """Run the HandLandmarker on an RGB frame; returns a list of hands, each
    a list of landmarks exposing .x/.y (normalized 0-1)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    return hand_landmarker.detect(mp_image).hand_landmarks


def preprocess_image(image_array: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Resize + normalize a BGR/RGB hand crop into the model's input format."""
    image = cv2.resize(image_array, target_size)
    image = image.astype(np.float32) / 127.5 - 1
    return np.expand_dims(image, axis=0)


def hand_bounding_box(hand_landmarks, frame_shape, padding: int = 20):
    """Pixel-space (x_min, y_min, x_max, y_max) bounding box for one hand's
    landmarks, clamped to the frame and padded by `padding` pixels."""
    h, w = frame_shape[:2]
    x_coords = [int(landmark.x * w) for landmark in hand_landmarks]
    y_coords = [int(landmark.y * h) for landmark in hand_landmarks]
    x_min = max(0, min(x_coords) - padding)
    x_max = min(w, max(x_coords) + padding)
    y_min = max(0, min(y_coords) - padding)
    y_max = min(h, max(y_coords) + padding)
    return x_min, y_min, x_max, y_max


def predict_hand(model, class_names, frame, hand_landmarks, padding: int = 20,
                  target_size=(224, 224)):
    """Crop the hand region out of `frame` using `hand_landmarks`, classify
    it, and return (label, confidence, bbox), or None if the crop is empty.
    """
    bbox = hand_bounding_box(hand_landmarks, frame.shape, padding)
    x_min, y_min, x_max, y_max = bbox
    hand_region = frame[y_min:y_max, x_min:x_max]
    if hand_region.size == 0:
        return None

    input_data = preprocess_image(hand_region, target_size)
    prediction = model.predict(input_data, verbose=0)
    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index])
    return label_for(class_names, index), confidence, bbox
