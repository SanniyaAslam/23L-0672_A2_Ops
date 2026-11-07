"""Task 1.2 - dataset audit. Usage: python src/audit.py data/raw --ai-key aiart"""
import argparse, hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ai-key", default="aiart", help="text in the path that marks the AI class")
    ap.add_argument("--out", default="data/audit.csv")
    a = ap.parse_args()

    rows = []
    for p in Path(a.root).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXT:
            continue
        r = {"path": str(p), "class": "ai" if a.ai_key in str(p).lower() else "real",
             "file_size": p.stat().st_size, "sha256": sha256(p), "ok": True}
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                r.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
        except Exception as e:
            r.update(ok=False, error=str(e)[:80])
        rows.append(r)

    df = pd.DataFrame(rows)
    df["duplicate"] = df.duplicated("sha256", keep="first")
    df.to_csv(a.out, index=False)

    good = df[df.ok]
    print("\nImages per class:\n", df["class"].value_counts())
    print("\nCorrupt files:", int((~df.ok).sum()), "| Exact duplicates:", int(df.duplicate.sum()))
    print("Cross-class duplicate hashes:", int((df.groupby("sha256")["class"].nunique() > 1).sum()))
    print("\nmin/median/max per class:")
    print(good.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"]).T)
    print("\nColour modes:\n", good["mode"].value_counts(), "\nFormats:\n", good["format"].value_counts())

if __name__ == "__main__":
    main()
