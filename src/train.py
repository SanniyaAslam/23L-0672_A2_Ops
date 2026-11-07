"""Tasks 2.3, 2.4, 3.3, 3.4, 3.5 - train and track.
  python src/train.py --model cnn --run-name 01_cnn_scratch
  python src/train.py --model transfer --run-name 02_mnv2 --lr 1e-3
  python src/train.py --model transfer --run-name 07_mnv2_autolog --autolog
"""
import argparse, os, re, subprocess
import numpy as np, mlflow
from dotenv import load_dotenv
from tensorflow import keras
from data import set_seeds, make_ds, SEED
from models import build_cnn, build_transfer, unfreeze_last_block
import evaluate as ev

load_dotenv()

class MLflowEpochLogger(keras.callbacks.Callback):
    """Manual logging (3.3a): logs exactly the per-epoch metrics we choose."""
    def __init__(self):
        super().__init__(); self.offset = 0
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=self.offset + epoch)

def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception: return "unknown"

def dvc_version():
    try: return re.search(r"md5: (\S+)", open("data/raw.dvc").read()).group(1)
    except Exception: return "unknown"

def callbacks(ckpt, logger):
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
           keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7)]
    return cbs + ([logger] if logger else [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "transfer"], default="transfer")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-finetune", action="store_true")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--autolog", action="store_true")
    a = ap.parse_args()

    set_seeds()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(f"{os.environ.get('ROLL_NUMBER', 'ROLL')}_A2_Ops")
    if a.autolog:
        mlflow.tensorflow.autolog(log_models=False)

    train = make_ds("train", a.img_size, a.batch, train=True, augment=not a.no_aug)
    val = make_ds("val", a.img_size, a.batch)
    test = make_ds("test", a.img_size, a.batch)
    os.makedirs("artifacts", exist_ok=True)
    ckpt = f"artifacts/{a.run_name}.keras"

    with mlflow.start_run(run_name=a.run_name):
        mlflow.set_tags({"git_commit": git_commit(), "dvc_data_version": dvc_version(), "seed": SEED,
                         "logging_mode": "autolog" if a.autolog else "manual",
                         "backbone": "mobilenetv2" if a.model == "transfer" else "scratch_cnn"})
        mlflow.log_params(vars(a))
        logger = None if a.autolog else MLflowEpochLogger()

        if a.model == "cnn":
            model = build_cnn(a.img_size, a.dropout, a.lr)
            h = model.fit(train, validation_data=val, epochs=a.epochs, callbacks=callbacks(ckpt, logger))
            hist = list(h.history["val_loss"])
        else:
            model = build_transfer(a.img_size, a.dropout, a.lr)
            h1 = model.fit(train, validation_data=val, epochs=a.epochs, callbacks=callbacks(ckpt, logger))
            hist = list(h1.history["val_loss"])
            if not a.no_finetune:
                trainable = unfreeze_last_block(model, a.ft_lr)
                print("Trainable backbone layers (stage 2):", trainable)
                mlflow.log_text("\n".join(trainable), "trainable_layers_stage2.txt")
                if logger: logger.offset = len(hist)
                h2 = model.fit(train, validation_data=val, epochs=a.ft_epochs,
                               callbacks=callbacks(ckpt, logger))
                hist += list(h2.history["val_loss"])

        mlflow.log_param("best_epoch", int(np.argmin(hist)) + 1)
        model = keras.models.load_model(ckpt)                 # best checkpoint by val_loss

        pv, yv = ev.predict(model, val), ev.labels("val")
        thr = ev.choose_threshold(yv, pv, a.beta)             # threshold from VALIDATION
        pt, yt = ev.predict(model, test), ev.labels("test")   # test scored once
        m = ev.metrics(yt, pt, thr)
        mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
        mlflow.log_metric("threshold", thr)
        print("threshold", thr, m)

        cm, roc = ev.plots(yt, pt, thr, a.run_name)
        errs = ev.misclassified_grid("test", pt, thr, f"artifacts/errors_{a.run_name}.png")
        for f in (cm, roc, errs): mlflow.log_artifact(f, "plots")
        for s in ("train", "val", "test"): mlflow.log_artifact(f"data/splits/{s}.csv", "splits")
        os.system("pip freeze > artifacts/requirements_freeze.txt")
        mlflow.log_artifact("artifacts/requirements_freeze.txt", "env")
        mlflow.log_artifact(ckpt, "keras_file")               # the file the cloud API will serve
        try:
            mlflow.keras.log_model(model, "model")            # registry-friendly flavour
        except Exception as e:
            print("mlflow.keras.log_model failed, falling back:", e)
            mlflow.tensorflow.log_model(model, "model")

if __name__ == "__main__":
    main()
