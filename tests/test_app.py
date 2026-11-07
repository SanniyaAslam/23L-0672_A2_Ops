import io, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from PIL import Image
from app import create_app
from db import Store

class FakeClf:
    def classify(self, name, data, mt): return "Real", 0.91, "test-model", "1"

def client():
    return create_app(Store("sqlite://"), FakeClf()).test_client()

def png():
    b = io.BytesIO(); Image.new("RGB", (8, 8)).save(b, "PNG"); b.seek(0); return b

def test_health():
    assert client().get("/health").json["status"] == "ok"

def test_predict_and_history():
    c = client()
    r = c.post("/predict", data={"image": (png(), "a.png")}, content_type="multipart/form-data")
    assert r.status_code == 200 and b"Real" in r.data
    assert b"test-model" in c.get("/history").data

def test_rejects_wrong_type():
    r = client().post("/predict", data={"image": (io.BytesIO(b"x"), "a.txt")},
                      content_type="multipart/form-data")
    assert r.status_code == 400

def test_rejects_fake_image():
    r = client().post("/predict", data={"image": (io.BytesIO(b"not an image"), "a.png")},
                      content_type="multipart/form-data")
    assert r.status_code == 422
