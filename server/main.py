"""
FastAPI/WebSocket backend for HandTalk's experimental `/temp` alt video-chat
page.

Exposes:
    GET /health         - Service/model readiness (for humans and the
                           Docker healthcheck).
    WS  /ws/{client_id}  - Receives video frames over a WebSocket, runs hand
                            detection + ASL prediction, and returns the
                            annotated frame and prediction to the target
                            peer.
"""

import os
import time
import json
import base64
import binascii
import logging
import traceback
from io import BytesIO
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from asl_model import load_model, load_labels, create_hand_landmarker, detect_hands, predict_hand

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on your CPU cores

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML components
MODEL_PATH = os.getenv("MODEL_PATH", "./best_model.keras")
LABELS_PATH = os.getenv("LABELS_PATH", "./labels.txt")
HAND_MODEL_PATH = os.getenv("HAND_MODEL_PATH", "./hand_landmarker.task")
PORT = int(os.getenv("PORT", "8000"))

START_TIME = time.time()

try:
    import tensorflow as tf

    model = load_model(MODEL_PATH)

    # Enable GPU memory growth to prevent OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    class_names = load_labels(LABELS_PATH)
    # Reduce to 1 hand for better performance
    hand_landmarker = create_hand_landmarker(HAND_MODEL_PATH, num_hands=1)

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())
    raise

connections: Dict[str, WebSocket] = {}


def process_frame(frame_data):
    """Process a frame with hand detection and ASL prediction."""
    try:
        if not isinstance(frame_data, str) or ',' not in frame_data:
            logger.warning("Rejected malformed frame payload (not a data URL)")
            return None

        img_data = base64.b64decode(frame_data.split(',')[1])
        img_array = np.array(Image.open(BytesIO(img_data)))

        # Reduce frame size for processing
        scale_factor = 0.5
        frame = cv2.resize(img_array, None, fx=scale_factor, fy=scale_factor)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process frame with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_landmarks = detect_hands(hand_landmarker, rgb_frame)

        detected_class = ""
        detected_confidence = 0.0

        for hand_landmarks in hands_landmarks:
            result = predict_hand(model, class_names, frame, hand_landmarks,
                                   padding=10, target_size=(160, 160))
            if result is None:
                continue
            detected_class, detected_confidence, (x_min, y_min, x_max, y_max) = result

            # Simplified visualization
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv2.putText(frame, f"{detected_class}",
                        (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Compress frame with lower quality for faster transmission
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        processed_frame = base64.b64encode(buffer).decode('utf-8')

        return {
            'frame': f'data:image/jpeg;base64,{processed_frame}',
            'prediction': {
                'class': detected_class,
                'confidence': detected_confidence
            }
        }

    except (binascii.Error, UnidentifiedImageError, ValueError) as e:
        logger.warning(f"Rejected bad frame payload: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        logger.error(traceback.format_exc())
        return None


@app.get("/health")
def health():
    """Report whether the model, labels, and hand-landmarker are ready, plus
    how many WebSocket clients are currently connected."""
    ready = model is not None and bool(class_names) and hand_landmarker is not None
    body = {
        "status": "ok" if ready else "degraded",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
        "hand_landmarker_ready": hand_landmarker is not None,
        "active_connections": len(connections),
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }
    return body


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connections[client_id] = websocket

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                logger.warning(f"Ignoring non-JSON message from {client_id}")
                continue

            if "target" not in message:
                logger.warning(f"Ignoring message from {client_id} with no target")
                continue

            if message.get("type") == "video-frame":
                # Process frame in thread pool
                future = executor.submit(process_frame, message.get("frame"))
                processed_data = future.result()

                if processed_data and message["target"] in connections:
                    response = {
                        "type": "processed-frame",
                        "frame": processed_data["frame"],
                        "prediction": processed_data["prediction"],
                        "from": client_id
                    }
                    await connections[message["target"]].send_text(json.dumps(response))
            else:
                # Handle other message types directly
                if message["target"] in connections:
                    await connections[message["target"]].send_text(data)

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        logger.error(traceback.format_exc())

    finally:
        if client_id in connections:
            del connections[client_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
