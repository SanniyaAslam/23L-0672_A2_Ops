"""Image preprocessing for the Vercel API - pure NumPy + Pillow (no TensorFlow).
It reproduces the training pipeline: decode to 3 colour channels, bilinear resize to size x size
(same method as tf.image.resize: half-pixel centres, no anti-aliasing), float values 0..255.
The model itself contains the rescaling layer, so no further scaling happens here."""
import io
import numpy as np
from PIL import Image


def resize_bilinear(img, out_h, out_w):
    h, w, _ = img.shape
    ys = np.clip((np.arange(out_h) + 0.5) * (h / out_h) - 0.5, 0, h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * (w / out_w) - 0.5, 0, w - 1)
    y0 = np.floor(ys).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    rows0, rows1 = img[y0], img[y1]
    top = rows0[:, x0] * (1 - wx) + rows0[:, x1] * wx
    bottom = rows1[:, x0] * (1 - wx) + rows1[:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def preprocess(data, size):
    img = Image.open(io.BytesIO(data)).convert("RGB")      # drops alpha, expands palettes, grey -> RGB
    return resize_bilinear(np.asarray(img, dtype=np.float32), size, size)
