# training/export_tflite.py

import os
import tensorflow as tf
import pandas as pd

MODELS_DIR = "../models/"
DATA_DIR = "../data/"
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
Keras_MODEL_PATH = os.path.join(MODELS_DIR, "trained_model.keras")

FLOAT32_TFLITE_PATH = os.path.join(MODELS_DIR, "edgeguard_float32.tflite")
FP16_TFLITE_PATH = os.path.join(MODELS_DIR, "edgeguard_fp16.tflite")
LABELS_TXT_PATH = os.path.join(MODELS_DIR, "labels.txt")


def build_labels_file():
    """
    Reads labels.csv and writes a labels.txt file with one class per line.
    The order must match the order used during training
    (we sorted unique labels: ['no_person', 'person']).
    """
    df = pd.read_csv(LABELS_CSV)
    class_names = sorted(df["label"].unique())

    with open(LABELS_TXT_PATH, "w") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"[INFO] labels.txt written to {LABELS_TXT_PATH}")
    print(f"[INFO] Classes (in order): {class_names}")
    return class_names


def export_float32(model):
    """
    Standard TFLite model (float32). Larger but easiest to debug.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(FLOAT32_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"[INFO] Float32 TFLite model saved to {FLOAT32_TFLITE_PATH}")


def export_fp16(model):
    """
    Float16 quantized TFLite model.
    Smaller & faster, great for Raspberry Pi.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model_fp16 = converter.convert()

    with open(FP16_TFLITE_PATH, "wb") as f:
        f.write(tflite_model_fp16)

    print(f"[INFO] Float16 TFLite model saved to {FP16_TFLITE_PATH}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[INFO] Loading trained Keras model...")
    model = tf.keras.models.load_model(Keras_MODEL_PATH, compile=False)
    print("[INFO] Model loaded.")

    print("[INFO] Building labels.txt...")
    build_labels_file()

    print("[INFO] Exporting Float32 TFLite model...")
    export_float32(model)

    print("[INFO] Exporting Float16 (quantized) TFLite model...")
    export_fp16(model)

    print("[INFO] All exports complete. Models are ready for Raspberry Pi.")


if __name__ == "__main__":
    main()
