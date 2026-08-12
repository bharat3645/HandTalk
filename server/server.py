"""
Flask backend for HandTalk's real-time ASL video-call feature.

Exposes:
    GET  /health       - Service/model/camera readiness (for humans, load
                          balancers, and the Docker healthcheck).
    GET  /video_feed    - MJPEG stream of the local webcam with hand
                           detection + ASL prediction overlay.
    GET  /prediction    - Latest prediction produced by /video_feed.
    POST /predict       - Predicts the ASL sign for a single base64-encoded
                           frame (used by the video-call page, which sends
                           frames captured from the browser instead of a
                           local webcam).
"""

import os
import time
import base64
import binascii
import logging
import traceback

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from asl_model import (
    load_model,
    load_labels,
    label_for,
    create_hand_landmarker,
    detect_hands,
    preprocess_image,
    predict_hand,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
# Reject oversized request bodies (e.g. a malformed/huge base64 payload)
# instead of buffering them into memory.
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB

MODEL_PATH = os.getenv("MODEL_PATH", "./best_model.keras")
LABELS_PATH = os.getenv("LABELS_PATH", "./labels.txt")
HAND_MODEL_PATH = os.getenv("HAND_MODEL_PATH", "./hand_landmarker.task")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
PORT = int(os.getenv("PORT", "5000"))

START_TIME = time.time()

try:
    model = load_model(MODEL_PATH)
    class_names = load_labels(LABELS_PATH)
    hand_landmarker = create_hand_landmarker(HAND_MODEL_PATH, num_hands=2)

    logger.info("Starting video capture...")
    # No explicit backend is passed so OpenCV picks the right one for the
    # host OS (CAP_V4L2 is Linux-only and breaks camera capture elsewhere).
    camera = cv2.VideoCapture(CAMERA_INDEX)

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())
    raise

latest_prediction = {"class": "", "confidence": 0.0}


def process_base64_image(base64_string):
    """Convert a base64-encoded image to OpenCV format, or None if it isn't
    valid base64 / decodable image data."""
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    try:
        img_bytes = base64.b64decode(base64_string, validate=True)
    except (binascii.Error, ValueError):
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    if nparr.size == 0:
        return None
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def generate_frames():
    """Generate video frames with hand detection and ASL prediction."""
    global latest_prediction

    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_landmarks = detect_hands(hand_landmarker, rgb_frame)

        detected_class = ""
        detected_confidence = 0.0

        for hand_landmarks in hands_landmarks:
            result = predict_hand(model, class_names, frame, hand_landmarks)
            if result is None:
                continue
            detected_class, detected_confidence, (x_min, y_min, x_max, y_max) = result

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{detected_class} ({detected_confidence*100:.2f}%)",
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Update latest prediction
        latest_prediction = {"class": detected_class, "confidence": detected_confidence}

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/health')
def health():
    """Report whether the model, labels, hand-landmarker, and camera are
    ready. Used by `docker-compose`'s healthcheck and for manual debugging
    of "why is nothing being detected" reports."""
    camera_open = bool(camera is not None and camera.isOpened())
    ready = model is not None and bool(class_names) and hand_landmarker is not None
    return jsonify({
        "status": "ok" if ready else "degraded",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
        "hand_landmarker_ready": hand_landmarker is not None,
        "camera_open": camera_open,
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }), 200 if ready else 503


@app.route('/video_feed')
def video_feed():
    """Stream video with hand detection and ASL prediction overlay."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prediction', methods=['GET'])
def get_prediction():
    """Return the latest ASL prediction."""
    return jsonify(latest_prediction), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        if not isinstance(data['image'], str) or not data['image']:
            return jsonify({'error': 'image must be a non-empty base64 string'}), 400

        image = process_base64_image(data['image'])
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_landmarks = detect_hands(hand_landmarker, rgb_image)

        for hand_landmarks in hands_landmarks:
            result = predict_hand(model, class_names, image, hand_landmarks)
            if result is None:
                continue
            label, confidence, _ = result
            return jsonify({'class': label, 'confidence': confidence})

        return jsonify({'class': '', 'confidence': 0.0}), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=PORT, debug=debug)
