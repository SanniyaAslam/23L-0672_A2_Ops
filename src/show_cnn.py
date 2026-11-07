import pandas as pd
from models import build_cnn
from data import set_seeds, make_ds

set_seeds()
model = build_cnn(img=224, dropout=0.4, lr=1e-3)
model.summary()

rows = []
for layer in model.layers:
    try:
        shape = str(tuple(layer.output.shape))
    except Exception:
        shape = "n/a"
    rows.append({"layer": layer.name, "type": layer.__class__.__name__,
                 "output_shape": shape, "params": layer.count_params()})
table = pd.DataFrame(rows)
print()
print(table.to_string(index=False))
print("Total params:", model.count_params())
table.to_csv("artifacts/cnn_layers.csv", index=False)

# sanity check: one forward pass on 8 real validation images
x, y = next(iter(make_ds("val", 224, 8)))
p = model.predict(x, verbose=0)
print("\nSanity check -> output shape:", p.shape,
      "| min:", round(float(p.min()), 3), "| max:", round(float(p.max()), 3))