"""Task 2.7 toy: Keras Tuner on Fashion-MNIST (learning rate + units)."""
import keras_tuner as kt, tensorflow as tf
(x, y), _ = tf.keras.datasets.fashion_mnist.load_data(); x = x[:10000] / 255.0; y = y[:10000]
def build(hp):
    m = tf.keras.Sequential([tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
              "sparse_categorical_crossentropy", ["accuracy"]); return m
t = kt.RandomSearch(build, objective="val_accuracy", max_trials=6, directory="artifacts/toy_tuner", project_name="fm")
t.search(x, y, epochs=3, validation_split=0.2); t.results_summary()
