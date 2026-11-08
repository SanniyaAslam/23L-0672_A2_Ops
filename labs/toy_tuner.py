"""Task 2.7 toy part: Keras Tuner (random search) on Fashion-MNIST.
Searches the learning rate and the number of units.   Run:  python labs/toy_tuner.py"""
import pandas as pd
import keras_tuner as kt
import tensorflow as tf

(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x = x[:10000].astype("float32") / 255.0
y = y[:10000]

def build(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=6, seed=42, overwrite=True,
                        directory="artifacts/toy_tuner", project_name="fm")
tuner.search(x, y, epochs=3, validation_split=0.2)

rows = []
for t in tuner.oracle.get_best_trials(num_trials=6):
    rows.append({"trial": t.trial_id, "units": t.hyperparameters.get("units"),
                 "lr": round(t.hyperparameters.get("lr"), 5), "val_accuracy": round(t.score, 4)})
print("\nTOY TUNER RESULTS (best first):")
print(pd.DataFrame(rows).to_string(index=False))
print("BEST SETTINGS:", tuner.get_best_hyperparameters(1)[0].values)
