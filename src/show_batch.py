import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import make_ds, set_seeds

set_seeds()
train = make_ds("train", img_size=224, batch=32, train=True, augment=True)
val = make_ds("val", img_size=224, batch=32)

x, y = next(iter(train))
print("train batch:", x.shape, x.dtype, "| pixel range:", float(x.numpy().min()), "to", float(x.numpy().max()))
xv, yv = next(iter(val))
print("val batch  :", xv.shape, xv.dtype)

fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for i, a in enumerate(ax.ravel()):
    a.imshow(x[i].numpy().clip(0, 255).astype("uint8"))
    a.set_title("AI-generated" if y[i] == 1 else "real")
    a.axis("off")
plt.tight_layout()
plt.savefig("artifacts/augmented_batch.png", dpi=120)
print("saved artifacts/augmented_batch.png")