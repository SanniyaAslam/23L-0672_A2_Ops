"""Task 1.3 - visual EDA. Usage: python src/eda.py"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("data/audit.csv")
df = df[df.ok & ~df.duplicate]

# Figure 1: 6 random samples per class
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for r, cls in enumerate(["real", "ai"]):
    paths = df[df["class"] == cls].sample(6, random_state=0).path
    for c, path in enumerate(paths):
        axes[r, c].imshow(Image.open(path).convert("RGB"))
        axes[r, c].axis("off")
    axes[r, 0].set_title(cls.upper(), loc="left", fontsize=14)
plt.tight_layout()
plt.savefig("artifacts/eda_samples.png", dpi=120)
plt.close()

# Figures 2-4: histograms per class (top 1% cut so outliers don't squash the plot)
for col in ["width", "height", "file_size"]:
    plt.figure(figsize=(7, 4))
    hi = df[col].quantile(0.99)
    for cls, g in df.groupby("class"):
        plt.hist(g[col][g[col] <= hi], bins=40, alpha=0.5, label=cls)
    plt.xlabel(col + " (top 1% cut)")
    plt.ylabel("number of images")
    plt.title(col + " per class")
    plt.legend()
    plt.savefig(f"artifacts/eda_{col}.png", dpi=120, bbox_inches="tight")
    plt.close()

print("Saved 4 figures in the artifacts folder.")