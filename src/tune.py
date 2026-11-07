"""Task 2.7 (apply): Keras Tuner on VisionGuard, 8 trials."""
import keras_tuner as kt
from tensorflow import keras
from data import make_ds, set_seeds
from models import build_transfer, unfreeze_last_block

def build(hp):
    m = build_transfer(224, dropout=hp.Float("dropout", 0.1, 0.6, step=0.1), lr=1e-3)
    unfreeze_last_block(m, hp.Float("lr", 1e-5, 1e-3, sampling="log"),
                        last_n=hp.Choice("unfrozen_layers", [0, 10, 30]))
    return m

if __name__ == "__main__":
    set_seeds()
    tuner = kt.RandomSearch(build, objective="val_loss", max_trials=8, seed=42,
                            directory="artifacts/tuner", project_name="visionguard")
    tuner.search(make_ds("train", train=True), validation_data=make_ds("val"), epochs=4,
                 callbacks=[keras.callbacks.EarlyStopping(patience=2)])
    tuner.results_summary()
    print(tuner.get_best_hyperparameters(1)[0].values)
