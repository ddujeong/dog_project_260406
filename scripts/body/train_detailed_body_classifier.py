import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "aihub_body", "detailed_body_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "body")
MODEL_PATH = os.path.join(MODEL_DIR, "detailed_body_classifier.keras")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


# 라벨
LABEL_MAP = {
    "slim": 0,
    "standard": 1,
    "broad": 2,
    "overweight": 3,
}
INDEX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_dataset():
    df = pd.read_csv(CSV_PATH)
    df = df[df["detailed_body_label"].isin(LABEL_MAP.keys())].copy()
    df["label_idx"] = df["detailed_body_label"].map(LABEL_MAP)
    return df.reset_index(drop=True)


def split_dataset(df):
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label_idx"],
        random_state=SEED
    )
    return train_df, val_df


def load_and_preprocess_image(path, label):
    img = load_img(path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    return img.astype("float32"), np.int32(label)


def make_dataset(df, training=True):
    paths = df["image_path"].values
    labels = df["label_idx"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(path, label):
        img, lab = tf.numpy_function(
            lambda p, l: load_and_preprocess_image(p.decode("utf-8"), l),
            [path, label],
            [tf.float32, tf.int32]
        )
        img.set_shape((224, 224, 3))
        lab.set_shape(())
        return img, lab

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000, seed=SEED)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(len(LABEL_MAP), activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_class_weight(df):
    y = df["label_idx"].values
    classes = np.unique(y)

    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def evaluate(model, val_df):
    val_ds = make_dataset(val_df, training=False)

    y_true = val_df["label_idx"].values
    y_pred = np.argmax(model.predict(val_ds), axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=INDEX_TO_LABEL.values()))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_dataset()
    train_df, val_df = split_dataset(df)

    print("=== Train Distribution ===")
    print(train_df["detailed_body_label"].value_counts())

    print("\n=== Val Distribution ===")
    print(val_df["detailed_body_label"].value_counts())

    train_ds = make_dataset(train_df)
    val_ds = make_dataset(val_df, training=False)

    model = build_model()
    class_weight = get_class_weight(train_df)

    print("\n=== Class Weight ===")
    print(class_weight)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        verbose=1
    )

    evaluate(model, val_df)

    model.save(MODEL_PATH)
    print(f"\n모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()