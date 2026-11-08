"""Tasks 2.5 / 3.5 - side-by-side comparison of your runs (read from MLflow / DagsHub).
  python src/metrics_table.py
Columns:
  epochs_logged  how many epochs of val_loss reached MLflow for that run
  best_val_loss  lowest validation loss over the logged epochs (threshold-free, used to pick the best run)
  best_val_auc   highest validation ROC-AUC over the logged epochs (threshold-free)
  threshold..roc_auc  TEST-set metrics at the threshold chosen on the validation set"""
import os
import mlflow, pandas as pd
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
load_dotenv()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
client = MlflowClient()
exp = mlflow.get_experiment_by_name(f"{os.environ.get('ROLL_NUMBER', 'ROLL')}_A2_Ops")
df = mlflow.search_runs([exp.experiment_id])
df = df[df["tags.mlflow.runName"].str.match(r"^\d", na=False)]       # only runs named 01_..., 02_...

def val_history(run_id, key):
    return [m.value for m in client.get_metric_history(run_id, key)]

rows = []
for _, r in df.iterrows():
    vl, va = val_history(r["run_id"], "val_loss"), val_history(r["run_id"], "val_auc")
    rows.append({"run": r["tags.mlflow.runName"],
                 "epochs_logged": len(vl),
                 "best_val_loss": min(vl) if vl else None,
                 "best_val_auc": max(va) if va else None,
                 "threshold": r.get("metrics.threshold"),
                 "accuracy": r.get("metrics.test_accuracy"),
                 "precision": r.get("metrics.test_precision"),
                 "recall": r.get("metrics.test_recall"),
                 "f1": r.get("metrics.test_f1"),
                 "roc_auc": r.get("metrics.test_roc_auc")})
table = pd.DataFrame(rows).sort_values("run")
num = [c for c in table.columns if c not in ("run", "epochs_logged")]
table[num] = table[num].astype(float).round(3)
pd.set_option("display.width", 220)
print(table.to_string(index=False))
table.to_csv("artifacts/run_comparison.csv", index=False)