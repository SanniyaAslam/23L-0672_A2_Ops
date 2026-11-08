"""Task 3.1 / 3.2 toy part: run twice with different learning rates.
  Local MLflow : python labs/toy_mlflow.py 0.001 ; python labs/toy_mlflow.py 0.01 ; then: mlflow ui
  Against DagsHub: same commands, with the MLFLOW_* values set in .env"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()
lr = float(sys.argv[1])
target = sys.argv[2] if len(sys.argv) > 2 else "local"      # "local" or "dagshub"
if target == "dagshub":
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
else:
    mlflow.set_tracking_uri("file:./mlruns_toy")
mlflow.set_experiment(f"toy_fashion_mnist_{target}")

(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x = x[:10000].astype("float32") / 255.0
y = y[:10000]
tf.keras.utils.set_random_seed(42)
model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(64, activation="relu"),
                             tf.keras.layers.Dense(10, activation="softmax")])
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

with mlflow.start_run(run_name=f"toy_lr_{lr}"):
    mlflow.set_tag("purpose", "toy")
    mlflow.log_param("lr", lr)
    mlflow.log_param("units", 64)
    hist = model.fit(x, y, epochs=5, validation_split=0.2, verbose=0)
    for e in range(5):
        mlflow.log_metric("loss", hist.history["loss"][e], step=e)
        mlflow.log_metric("val_accuracy", hist.history["val_accuracy"][e], step=e)
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="val loss")
    plt.legend(); plt.title(f"toy run, lr={lr}")
    plt.savefig("toy_loss.png"); plt.close()
    mlflow.log_artifact("toy_loss.png")
    mlflow.keras.log_model(model, "model")
    print(f"logged run toy_lr_{lr} to {mlflow.get_tracking_uri()} | final val_accuracy:",
          round(hist.history["val_accuracy"][-1], 4))
