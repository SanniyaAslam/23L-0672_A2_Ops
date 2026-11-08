"""Re-score your saved models with a better decision-threshold rule (NO retraining).
Rule: on the VALIDATION set, take the highest threshold that still catches at least
--target-recall of the AI images (missing a fake is the worse error), so as few real photos as
possible are wrongly flagged. The test set is only used to report the result.
  python src/reevaluate.py"""
import argparse, os
import numpy as np, pandas as pd, mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from data import make_ds
import evaluate as ev
load_dotenv()

def pick_threshold(y, p, target):
    ts = np.linspace(0.01, 0.99, 99)
    ok = [t for t in ts if (p[y == 1] >= t).mean() >= target]
    return float(max(ok)) if ok else float(ts[0])

def history(client, run_id, key):
    return [m.value for m in client.get_metric_history(run_id, key)]

ap = argparse.ArgumentParser()
ap.add_argument("--target-recall", type=float, default=0.90)
a = ap.parse_args()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
client = MlflowClient()
exp = mlflow.get_experiment_by_name(f"{os.environ.get('ROLL_NUMBER', 'ROLL')}_A2_Ops")
runs = mlflow.search_runs([exp.experiment_id])
runs = runs[runs["tags.mlflow.runName"].str.match(r"^\d", na=False)].sort_values("tags.mlflow.runName")

rows = []
for _, r in runs.iterrows():
    name, ckpt = r["tags.mlflow.runName"], f"artifacts/{r['tags.mlflow.runName']}.keras"
    if not os.path.exists(ckpt):
        print("skipping (no saved model file):", name); continue
    size = int(r["params.img_size"])
    model = keras.models.load_model(ckpt)
    pv, yv = ev.predict(model, make_ds("val", size, 32)), ev.labels("val")
    pt, yt = ev.predict(model, make_ds("test", size, 32)), ev.labels("test")
    thr = pick_threshold(yv, pv, a.target_recall)
    m = ev.metrics(yt, pt, thr)
    tn, fp, fn, tp = confusion_matrix(yt, (pt >= thr).astype(int)).ravel()
    m["specificity"] = tn / (tn + fp)
    cm_png, _ = ev.plots(yt, pt, thr, name + "_v2")
    with mlflow.start_run(run_id=r["run_id"]):
        mlflow.set_tag("threshold_rule_v2", f"highest threshold with validation recall >= {a.target_recall}")
        mlflow.log_metric("v2_threshold", thr)
        mlflow.log_metrics({f"v2_test_{k}": float(v) for k, v in m.items() if k != "roc_auc"})
        mlflow.log_artifact(cm_png, "plots")
    vl, va = history(client, r["run_id"], "val_loss"), history(client, r["run_id"], "val_auc")
    rows.append({"run": name, "best_val_loss": min(vl), "best_val_auc": max(va), "test_auc": m["roc_auc"],
                 "threshold": thr, "accuracy": m["accuracy"], "precision": m["precision"],
                 "recall": m["recall"], "specificity": m["specificity"], "f1": m["f1"]})

table = pd.DataFrame(rows)
num = [c for c in table.columns if c != "run"]
table[num] = table[num].astype(float).round(3)
pd.set_option("display.width", 220)
print(f"\nThreshold rule: highest validation threshold with recall >= {a.target_recall}")
print(table.to_string(index=False))
table.to_csv("artifacts/run_comparison_v2.csv", index=False)
