"""Task 3.1 / 3.2 toy: run twice with different LRs. Local: mlflow ui. DagsHub: set env vars first.
  python labs/toy_mlflow.py 0.001 ; python labs/toy_mlflow.py 0.01"""
import sys, os, mlflow, matplotlib.pyplot as plt
import tensorflow as tf
from dotenv import load_dotenv
load_dotenv()
lr = float(sys.argv[1])
if os.environ.get("MLFLOW_TRACKING_URI"): mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("toy_fashion_mnist")
(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x = x[:10000] / 255.0; y = y[:10000]
m = tf.keras.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation="relu"),
                         tf.keras.layers.Dense(10, activation="softmax")])
m.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", ["accuracy"])
with mlflow.start_run(run_name=f"toy_lr_{lr}"):
    mlflow.set_tag("purpose", "toy"); mlflow.log_param("lr", lr)
    h = m.fit(x, y, epochs=5, validation_split=0.2, verbose=0)
    for e in range(5):
        mlflow.log_metric("val_accuracy", h.history["val_accuracy"][e], step=e)
        mlflow.log_metric("loss", h.history["loss"][e], step=e)
    plt.plot(h.history["loss"]); plt.title(f"loss lr={lr}"); plt.savefig("toy_loss.png")
    mlflow.log_artifact("toy_loss.png")
    mlflow.keras.log_model(m, "model")
