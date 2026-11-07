"""Task 3.6 - load the model BY ALIAS and classify one image.
Also saves models/champion.keras: the file the cloud API serves, so no DagsHub call per request.
  python src/load_champion.py --image some.jpg"""
import argparse, os, shutil
import tensorflow as tf, mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--name", default="visionguard-champion"); ap.add_argument("--alias", default="champion")
ap.add_argument("--size", type=int, default=224)
a = ap.parse_args()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mv = MlflowClient().get_model_version_by_alias(a.name, a.alias)
print(f"{a.name}@{a.alias} -> version {mv.version}, run {mv.run_id}")
local = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras_file")
f = next(os.path.join(local, x) for x in os.listdir(local) if x.endswith(".keras"))
os.makedirs("models", exist_ok=True); shutil.copy(f, "models/champion.keras")
model = tf.keras.models.load_model("models/champion.keras")
img = tf.image.resize(tf.io.decode_image(tf.io.read_file(a.image), channels=3, expand_animations=False),
                      (a.size, a.size))
p = float(model.predict(img[None], verbose=0)[0, 0])
print("P(AI-generated) =", round(p, 4), "->", "AI-generated" if p >= 0.5 else "Real")
