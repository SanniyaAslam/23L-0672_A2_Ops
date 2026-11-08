"""Task 2.7 apply part: Keras Tuner on VisionGuard.
Searches learning rate, dropout and number of unfrozen backbone layers.
  python src/tune.py            (8 trials, 4 epochs each)"""
import argparse
import pandas as pd
import keras_tuner as kt
from tensorflow import keras
from data import make_ds, set_seeds
from models import build_transfer, unfreeze_last_block

def build(hp):
    model = build_transfer(224, dropout=hp.Float("dropout", 0.1, 0.6, step=0.1), lr=1e-3)
    unfreeze_last_block(model, hp.Float("lr", 1e-5, 1e-3, sampling="log"),
                        last_n=hp.Choice("unfrozen_layers", [0, 10, 30]))
    return model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()

    set_seeds()
    tuner = kt.RandomSearch(build, objective="val_loss", max_trials=a.trials, seed=42, overwrite=True,
                            directory="artifacts/tuner", project_name="visionguard")
    tuner.search(make_ds("train", train=True), validation_data=make_ds("val"), epochs=a.epochs,
                 callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

    rows = []
    for t in tuner.oracle.get_best_trials(num_trials=a.trials):
        hp = t.hyperparameters
        rows.append({"trial": t.trial_id, "dropout": round(hp.get("dropout"), 2),
                     "lr": round(hp.get("lr"), 6), "unfrozen_layers": hp.get("unfrozen_layers"),
                     "val_loss": round(t.score, 4)})
    table = pd.DataFrame(rows)
    table.to_csv("artifacts/tuner_results.csv", index=False)
    print("\nVISIONGUARD TUNER RESULTS (best first):")
    print(table.to_string(index=False))
    print("BEST SETTINGS:", tuner.get_best_hyperparameters(1)[0].values)
