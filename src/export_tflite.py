"""Task 7 - convert the champion Keras model to TensorFlow Lite (small enough for the free Vercel plan)
and PROVE the converted model gives the same answers as the original.
  python src/export_tflite.py"""
import contextlib
import io
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, "vercel_api")
from preprocess import preprocess

SRC = "models/champion.keras"
OUT = "vercel_api/model.tflite"
THRESHOLD = float(os.environ.get("THRESHOLD", 0.29))

model = tf.keras.models.load_model(SRC, compile=False)
size = int(model.input_shape[1])
with tempfile.TemporaryDirectory() as tmp:
    with contextlib.redirect_stdout(io.StringIO()):             # hide the very long export log
        model.export(tmp)                                       # Keras model -> SavedModel
    tflite = tf.lite.TFLiteConverter.from_saved_model(tmp).convert()
with open(OUT, "wb") as f:
    f.write(tflite)
print(f"Saved {OUT}: {len(tflite) / 1e6:.1f} MB (input size {size}x{size})")

interp = tf.lite.Interpreter(model_content=tflite)
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

paths = pd.read_csv("data/splits/test.csv").sample(40, random_state=0).path
diffs, agree = [], 0
for path in paths:
    with open(path, "rb") as f:
        data = f.read()
    img = tf.image.resize(tf.io.decode_image(data, channels=3, expand_animations=False), (size, size))
    p_keras = float(model.predict(img[None], verbose=0)[0, 0])          # original model + training pipeline
    interp.set_tensor(inp["index"], preprocess(data, size)[None].astype(np.float32))
    interp.invoke()
    p_lite = float(interp.get_tensor(out["index"])[0, 0])             # converted model + API pipeline
    diffs.append(abs(p_keras - p_lite))
    agree += int((p_keras >= THRESHOLD) == (p_lite >= THRESHOLD))
print(f"\nChecked {len(diffs)} test images (Keras + training pipeline vs TFLite + API pipeline):")
print(f"  largest difference in P(AI): {max(diffs):.5f} | average difference: {np.mean(diffs):.5f}")
print(f"  same verdict at threshold {THRESHOLD}: {agree} of {len(diffs)} images")
