"""Inference module (Task 4.1 and Task 8).

LocalClassifier  - loads the champion Keras model ONCE at start-up and runs it inside this process.
RemoteClassifier - sends the image to the model API deployed in the cloud (used from Task 8 on).
build_classifier() picks one from the environment variable INFERENCE_MODE ("local" or "remote").
"""
import os
import requests


class LocalClassifier:
    def __init__(self, model_path, img_size=224, threshold=0.5,
                 model_name="visionguard-champion", model_version="1"):
        import tensorflow as tf              # imported here so the slim web image (remote mode) needs no TensorFlow
        self.tf = tf
        self.model = tf.keras.models.load_model(model_path)    # loaded once, reused for every request
        self.size, self.threshold = int(img_size), float(threshold)
        self.model_name, self.model_version = model_name, str(model_version)

    def classify(self, filename, data: bytes, mimetype=None):
        tf = self.tf
        # same steps as the training pipeline: decode to 3 channels, bilinear resize, floats 0..255
        # (the model itself contains the Rescaling layer, so no extra scaling here)
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
        img = tf.image.resize(img, (self.size, self.size))
        p = float(self.model.predict(img[None], verbose=0)[0, 0])
        label = "AI-generated" if p >= self.threshold else "Real"
        confidence = p if p >= self.threshold else 1 - p
        return label, round(confidence, 4), self.model_name, self.model_version


class RemoteClassifier:
    def __init__(self, url, api_key, timeout=60):
        self.url, self.key, self.timeout = url, api_key, timeout

    def classify(self, filename, data: bytes, mimetype=None):
        r = requests.post(self.url, headers={"X-API-Key": self.key},
                          files={"image": (filename, data, mimetype or "application/octet-stream")},
                          timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        return j["label"], j["confidence"], j["model_name"], j["model_version"]


def build_classifier():
    if os.environ.get("INFERENCE_MODE", "remote") == "local":
        return LocalClassifier(model_path=os.environ.get("MODEL_PATH", "models/champion.keras"),
                               img_size=os.environ.get("IMG_SIZE", 224),
                               threshold=os.environ.get("THRESHOLD", 0.5),
                               model_name=os.environ.get("MODEL_NAME", "visionguard-champion"),
                               model_version=os.environ.get("MODEL_VERSION", "1"))
    return RemoteClassifier(os.environ.get("INFERENCE_API_URL", ""), os.environ.get("INFERENCE_API_KEY", ""))
