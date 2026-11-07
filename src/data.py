"""Split lists + tf.data pipeline (Task 2.1)."""
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
SPLIT_DIR = "data/splits"

def make_splits(audit_csv="data/audit.csv"):
    df = pd.read_csv(audit_csv)
    df = df[df.ok & ~df.duplicate].copy()            # drop corrupt + duplicates
    df["label"] = (df["class"] == "ai").astype(int)    # 1 = AI-generated
    df = df[["path", "label"]]
    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(f"{SPLIT_DIR}/{name}.csv", index=False)
        print(name, len(part), round(part.label.mean(), 3))

def set_seeds(seed=SEED):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)

def augmenter():
    L = tf.keras.layers
    return tf.keras.Sequential([L.RandomFlip("horizontal"), L.RandomRotation(0.05),
                                L.RandomZoom(0.1), L.RandomContrast(0.1)])

def make_ds(split, img_size=224, batch=32, train=False, augment=True):
    df = pd.read_csv(f"{SPLIT_DIR}/{split}.csv")
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values.astype("float32")))
    if train:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    def load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (img_size, img_size))   # squashes aspect ratio; float32, 0..255
        return img, y

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if train and augment:                                   # augmentation: TRAIN ONLY
        aug = augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    make_splits()