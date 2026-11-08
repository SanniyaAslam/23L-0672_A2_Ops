"""Task 4.1 - console proof that the inference module works: loads the model ONCE, then classifies
one real and one AI-generated test image and shows how long each step takes.
  python src/check_inference.py"""
import os, sys, time
import pandas as pd
sys.path.insert(0, "app")
from inference import LocalClassifier

t0 = time.time()
clf = LocalClassifier(model_path=os.environ.get("MODEL_PATH", "models/champion.keras"),
                      img_size=os.environ.get("IMG_SIZE", 224),
                      threshold=os.environ.get("THRESHOLD", 0.29),
                      model_version=os.environ.get("MODEL_VERSION", "2"))
print(f"Model loaded ONCE in {time.time() - t0:.1f} s (threshold {clf.threshold})")

df = pd.read_csv("data/splits/test.csv")
for label, name in [(0, "real"), (1, "AI-generated")]:
    path = df[df.label == label].path.iloc[1]
    with open(path, "rb") as f:
        data = f.read()
    t = time.time()
    out = clf.classify(os.path.basename(path), data)
    print(f"\nKnown {name} image: {path}")
    print(f"  -> verdict: {out[0]} | confidence: {out[1]} | model: {out[2]} v{out[3]} | took {time.time() - t:.2f} s")
