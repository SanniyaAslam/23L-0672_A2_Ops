"""Task 2.5 - side-by-side test metrics of your runs (read from MLflow / DagsHub).
  python src/metrics_table.py"""
import os
import mlflow, pandas as pd
from dotenv import load_dotenv
load_dotenv()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
exp = mlflow.get_experiment_by_name(f"{os.environ.get('ROLL_NUMBER', 'ROLL')}_A2_Ops")
df = mlflow.search_runs([exp.experiment_id])
df = df[df["tags.mlflow.runName"].str.match(r"^\d", na=False)]       # only runs named 01_..., 02_...

want = {"tags.mlflow.runName": "run", "params.best_epoch": "best_epoch", "metrics.threshold": "threshold",
        "metrics.test_accuracy": "accuracy", "metrics.test_precision": "precision",
        "metrics.test_recall": "recall", "metrics.test_f1": "f1", "metrics.test_roc_auc": "roc_auc"}
cols = [c for c in want if c in df.columns]
table = df[cols].rename(columns=want).sort_values("run")
num = [c for c in table.columns if c not in ("run", "best_epoch")]
table[num] = table[num].astype(float).round(3)
pd.set_option("display.width", 200)
print(table.to_string(index=False))
