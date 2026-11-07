"""Tasks 2.2 / 2.3. Preprocessing lives INSIDE the model, so serving == training."""
from tensorflow import keras
from tensorflow.keras import layers as L

def compile_model(m, lr):
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m

def build_cnn(img=224, dropout=0.4, lr=1e-3):
    inp = keras.Input((img, img, 3))
    x = L.Rescaling(1 / 255.0)(inp)                              # 0..255 -> 0..1
    for f in (32, 64, 128):
        x = L.Conv2D(f, 3, padding="same", use_bias=False)(x)     # learn local patterns
        x = L.BatchNormalization()(x)                             # stabilise training
        x = L.ReLU()(x)
        x = L.MaxPooling2D()(x)                                   # shrink, keep strongest response
    x = L.GlobalAveragePooling2D()(x)                             # far fewer params than Flatten
    x = L.Dropout(dropout)(x)                                     # regularise
    out = L.Dense(1, activation="sigmoid")(x)                     # P(AI-generated)
    return compile_model(keras.Model(inp, out, name="baseline_cnn"), lr)

def build_transfer(img=224, dropout=0.3, lr=1e-3):
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                          input_shape=(img, img, 3))
    base.trainable = False                                        # stage 1: frozen
    inp = keras.Input((img, img, 3))
    x = L.Rescaling(1 / 127.5, offset=-1)(inp)                    # same as mobilenet_v2.preprocess_input: [-1, 1]
    x = base(x, training=False)                                   # BatchNorm stays in inference mode
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(1, activation="sigmoid")(x)
    return compile_model(keras.Model(inp, out, name="mobilenetv2_transfer"), lr)

def get_backbone(model):
    return next(l for l in model.layers if isinstance(l, keras.Model))

def unfreeze_last_block(model, lr=1e-5, last_n=None):
    """Unfreeze only the last block (block_16 + Conv_1), or the last_n layers. BatchNorm stays frozen."""
    base = get_backbone(model)
    base.trainable = True
    for l in base.layers:
        l.trainable = False
    if last_n is None:
        targets = [l for l in base.layers if l.name.startswith(("block_16", "Conv_1"))]
    else:
        targets = base.layers[-last_n:] if last_n > 0 else []
    for l in targets:
        if not isinstance(l, L.BatchNormalization):
            l.trainable = True
    compile_model(model, lr)                                      # recompile after changing trainable
    return [l.name for l in base.layers if l.trainable]
