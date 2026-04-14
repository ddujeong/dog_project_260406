import os
import sys
import math
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


# 프로젝트 루트 경로 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "aihub_body", "body_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "body")
MODEL_PATH = os.path.join(MODEL_DIR, "body_classifier_efficientnetb0.keras")


LABEL_MAP = {
    "slim": 0,
    "normal": 1,
    "overweight": 2,
}
INDEX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_dataset():
    df = pd.read_csv(CSV_PATH)

    df = df[df["image_path"].notna()].copy()
    df = df[df["body_label"].isin(LABEL_MAP.keys())].copy()

    df["label_idx"] = df["body_label"].map(LABEL_MAP)
    df = df.reset_index(drop=True)

    return df


def split_dataset(df: pd.DataFrame):
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label_idx"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_and_preprocess_image(image_path: str, label_idx: int):
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)
    label = tf.one_hot(label_idx, depth=len(LABEL_MAP))
    return image, label


def make_dataset(df: pd.DataFrame, training: bool = True):
    image_paths = df["image_path"].values
    label_indices = df["label_idx"].values

    ds = tf.data.Dataset.from_tensor_slices((image_paths, label_indices))

    def _map_fn(path, label):
        image, one_hot = tf.numpy_function(
            func=lambda p, l: load_and_preprocess_image(
                p.decode("utf-8"), int(l)
            ),
            inp=[path, label],
            Tout=[tf.float32, tf.float32]
        )
        image.set_shape((224, 224, 3))
        one_hot.set_shape((len(LABEL_MAP),))
        return image, one_hot

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=min(len(df), 1000), seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(len(LABEL_MAP), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def get_class_weights(train_df: pd.DataFrame):
    y = train_df["label_idx"].values
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    return class_weight_dict


def evaluate_model(model, val_df: pd.DataFrame):
    val_ds = make_dataset(val_df, training=False)

    y_true = val_df["label_idx"].values
    y_pred_probs = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(
        y_true,
        y_pred,
        target_names=[INDEX_TO_LABEL[i] for i in range(len(LABEL_MAP))]
    ))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_dataset()
    train_df, val_df = split_dataset(df)

    print("=== Dataset Info ===")
    print(f"total: {len(df)}")
    print(f"train: {len(train_df)}")
    print(f"val: {len(val_df)}")

    print("\n=== Train Label Distribution ===")
    print(train_df["body_label"].value_counts())

    print("\n=== Val Label Distribution ===")
    print(val_df["body_label"].value_counts())

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)

    model, base_model = build_model()

    class_weight = get_class_weights(train_df)
    print("\n=== Class Weights ===")
    print(class_weight)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    print("\n=== Stage 1: feature extractor 학습 ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    print("\n=== Validation Evaluation ===")
    evaluate_model(model, val_df)

    model.save(MODEL_PATH)
    print(f"\n모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()