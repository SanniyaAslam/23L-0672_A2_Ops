"""Task 3.6 - load the model BY ALIAS and classify one image (a separate small script).
Also saves models/champion.keras: the file the cloud API serves, so no DagsHub call per request.
  python src/load_champion.py --image path/to/some.jpg"""
import argparse, os, shutil
import tensorflow as tf, mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
load_dotenv()

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--name", default="visionguard-champion")
ap.add_argument("--alias", default="champion")
a = ap.parse_args()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
client = MlflowClient()
mv = client.get_model_version_by_alias(a.name, a.alias)
run = client.get_run(mv.run_id)
size = int(run.data.params.get("img_size", 224))
thr = run.data.metrics.get("v2_threshold", 0.5)          # threshold chosen on validation data
print(f"{a.name}@{a.alias} -> version {mv.version}, run {run.data.tags.get('mlflow.runName')} ({mv.run_id})")

local = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras_file")
f = next(os.path.join(local, x) for x in os.listdir(local) if x.endswith(".keras"))
os.makedirs("models", exist_ok=True)
shutil.copy(f, "models/champion.keras")
model = tf.keras.models.load_model("models/champion.keras")

img = tf.image.resize(tf.io.decode_image(tf.io.read_file(a.image), channels=3, expand_animations=False),
                      (size, size))
p = float(model.predict(img[None], verbose=0)[0, 0])
label = "AI-generated" if p >= thr else "Real"
print(f"P(AI-generated) = {p:.4f} | threshold = {thr:.2f} -> {label}")