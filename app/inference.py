"""Inference client: the UI sends the image to the cloud model API (Task 8)."""
import requests

class RemoteClassifier:
    def __init__(self, url, api_key, timeout=60):
        self.url, self.key, self.timeout = url, api_key, timeout

    def classify(self, filename, data: bytes, mimetype):
        r = requests.post(self.url, headers={"X-API-Key": self.key},
                          files={"image": (filename, data, mimetype)}, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        return j["label"], j["confidence"], j["model_name"], j["model_version"]
