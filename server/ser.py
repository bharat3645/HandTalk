"""
Flask backend for HandTalk's self-testing / ASL practice page.

Exposes:
    GET /health      - Service/model/camera readiness (for humans, load
                        balancers, and the Docker healthcheck).
    GET /video_feed  - MJPEG stream of the local webcam with hand detection
                        + ASL prediction overlay.
    GET /prediction  - Latest prediction produced by /video_feed.
"""

import os
import time
import logging
import traceback

import cv2
from flask import Flask, Response, jsonify
from flask_cors import CORS

from asl_model import (
    load_model,
    load_labels,
    create_hand_landmarker,
    detect_hands,
    predict_hand,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.getenv("MODEL_PATH", "./best_model.keras")
LABELS_PATH = os.getenv("LABELS_PATH", "./labels.txt")
HAND_MODEL_PATH = os.getenv("HAND_MODEL_PATH", "./hand_landmarker.task")
# Defaults to camera index 1 so this can run alongside server.py (index 0)
# on machines with two cameras. Override with CAMERA_INDEX=0 if you only
# have a single webcam and are running this service on its own.
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "1"))
PORT = int(os.getenv("PORT", "5001"))

START_TIME = time.time()

try:
    model = load_model(MODEL_PATH)
    class_names = load_labels(LABELS_PATH)
    hand_landmarker = create_hand_landmarker(HAND_MODEL_PATH, num_hands=2)

    logger.info("Starting video capture...")
    camera = cv2.VideoCapture(CAMERA_INDEX)  # Open webcam

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())
    raise

latest_prediction = {"class": "", "confidence": 0.0}


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


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=PORT, debug=debug)
