"""VisionGuard model API for Vercel: Flask + TensorFlow Lite. The model is loaded once per instance."""
import hmac
import os
import threading

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, jsonify, request
from PIL import UnidentifiedImageError

from preprocess import preprocess

HERE = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.environ.get("INFERENCE_API_KEY", "")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.29"))
MODEL_NAME = os.environ.get("MODEL_NAME", "visionguard-champion")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "2")

interpreter = Interpreter(model_path=os.path.join(HERE, "model.tflite"))   # loaded once, reused
interpreter.allocate_tensors()
INPUT = interpreter.get_input_details()[0]
OUTPUT = interpreter.get_output_details()[0]
SIZE = int(INPUT["shape"][1])
lock = threading.Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024      # Vercel itself rejects bodies above 4.5 MB


def predict_probability(data):
    x = preprocess(data, SIZE)[None].astype(np.float32)
    with lock:
        interpreter.set_tensor(INPUT["index"], x)
        interpreter.invoke()
        return float(interpreter.get_tensor(OUTPUT["index"])[0, 0])


@app.get("/")
def index():
    return jsonify(service="VisionGuard model API", health="/health", predict="POST /predict")


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_NAME, version=MODEL_VERSION, input_size=SIZE)


@app.post("/predict")
def predict():
    key = request.headers.get("X-API-Key", "")
    if not API_KEY or not hmac.compare_digest(key, API_KEY):
        return jsonify(error="unauthorized"), 401
    f = request.files.get("image")
    if f is None:
        return jsonify(error="no image"), 400
    try:
        p = predict_probability(f.read())
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error="not a decodable image"), 422
    label = "AI-generated" if p >= THRESHOLD else "Real"
    confidence = p if p >= THRESHOLD else 1 - p
    return jsonify(label=label, confidence=round(confidence, 4), p_ai=round(p, 4),
                   model_name=MODEL_NAME, model_version=MODEL_VERSION)


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="image too large (limit 4 MB)"), 413
