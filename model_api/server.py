"""Model API deployed on the cloud VM. Loads the model ONCE, protected by an API key."""
import io, os
import numpy as np, tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError

IMG = int(os.environ.get("IMG_SIZE", 224))
API_KEY = os.environ["INFERENCE_API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "visionguard-champion")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))

model = tf.keras.models.load_model(os.environ.get("MODEL_PATH", "/app/models/champion.keras"))  # once
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 5)) * 1024 * 1024

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION}

@app.post("/predict")
def predict():
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="unauthorized"), 401
    f = request.files.get("image")
    if f is None:
        return jsonify(error="no image"), 400
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="not a decodable image"), 422
    x = tf.image.resize(np.asarray(img, dtype="float32"), (IMG, IMG))   # same op as training
    p = float(model.predict(x[None], verbose=0)[0, 0])
    label = "AI-generated" if p >= THRESHOLD else "Real"
    conf = p if p >= THRESHOLD else 1 - p
    return jsonify(label=label, confidence=round(conf, 4), p_ai=round(p, 4),
                   model_name=MODEL_NAME, model_version=MODEL_VERSION)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
