"""Task 3.6 - register a run as a model version. Run it twice (two runs), alias one 'champion'.
  python src/register_model.py --run-id <id> [--alias champion]"""
import argparse, os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True); ap.add_argument("--alias")
ap.add_argument("--name", default="visionguard-champion")
a = ap.parse_args()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mv = mlflow.register_model(f"runs:/{a.run_id}/model", a.name)
print("registered version", mv.version)
if a.alias:
    MlflowClient().set_registered_model_alias(a.name, a.alias, mv.version)
    print(f"alias '{a.alias}' -> v{mv.version}")
