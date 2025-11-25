# training/train_model.py

import os
import matplotlib.pyplot as plt
from dataset_loader import EdgeGuardDataset
from model_architecture import build_mobilenet_model
import tensorflow as tf

SAVE_DIR = "../models/"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_metrics(history):
    # Accuracy
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("../models/accuracy_plot.png")
    plt.close()

    # Loss
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("../models/loss_plot.png")
    plt.close()

def main():
    print("[INFO] Loading dataset...")
    dataset = EdgeGuardDataset()
    train_ds, val_ds, classes = dataset.prepare_dataset()

    print("[INFO] Classes:", classes)

    print("[INFO] Building model...")
    model = build_mobilenet_model(num_classes=len(classes))

    print("[INFO] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15
    )

    print("[INFO] Training complete.")

    # Save model
    model_path = os.path.join(SAVE_DIR, "trained_model.keras")
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    # Save plots
    plot_metrics(history)
    print("[INFO] Training curves saved.")

if __name__ == "__main__":
    main()
