"""Tasks 2.5 / 2.6. Threshold chosen on VALIDATION only; the test set is scored once."""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, fbeta_score)

def predict(model, ds):
    return model.predict(ds, verbose=0).ravel()

def labels(split):
    return pd.read_csv(f"data/splits/{split}.csv").label.values

def choose_threshold(y, p, beta=2.0):
    """Maximise F-beta on validation. beta>1 favours recall of the AI class (a missed fake is worse)."""
    ts = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y, p >= t, beta=beta) for t in ts]
    return float(ts[int(np.argmax(scores))])

def metrics(y, p, t):
    yh = (p >= t).astype(int)
    return {"accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh),
            "recall": recall_score(y, yh), "f1": f1_score(y, yh), "roc_auc": roc_auc_score(y, p)}

def plots(y, p, t, tag, out="artifacts"):
    cm = confusion_matrix(y, (p >= t).astype(int))
    plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["real", "ai"]); plt.yticks([0, 1], ["real", "ai"])
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"Confusion ({tag}, t={t:.2f})")
    f1 = f"{out}/confusion_{tag}.png"; plt.savefig(f1, bbox_inches="tight"); plt.close()
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(4, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({tag})")
    f2 = f"{out}/roc_{tag}.png"; plt.savefig(f2, bbox_inches="tight"); plt.close()
    return f1, f2

def misclassified_grid(split, p, t, out="artifacts/errors.png", n=10):
    from PIL import Image
    df = pd.read_csv(f"data/splits/{split}.csv"); df["p"] = p
    df["pred"] = (p >= t).astype(int)
    wrong = df[df.pred != df.label]
    bad = wrong.sample(min(n, len(wrong)), random_state=0)
    fig, ax = plt.subplots(2, 5, figsize=(16, 7))
    for a in ax.ravel(): a.axis("off")
    for a, (_, r) in zip(ax.ravel(), bad.iterrows()):
        a.imshow(Image.open(r.path).convert("RGB"))
        a.set_title(f"true={'ai' if r.label else 'real'}  p(ai)={r.p:.2f}", fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    return out
