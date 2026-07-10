import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATASET_PATH = "../website/dataset"
MODEL_SAVE_PATH = "models/meat_model.keras"
REPORT_SAVE_PATH = "models/training_report.json"
CONFUSION_MATRIX_PATH = "../website/uploads/confusion_matrix.png"

# Global training status
training_status = {
    "running": False,
    "epoch": 0,
    "total_epochs": 0,
    "accuracy": 0,
    "loss": 0,
    "val_accuracy": 0,
    "val_loss": 0,
    "message": "Idle",
    "done": False,
    "error": "",
    "epoch_logs": [],
    "confusion_matrix_image": ""
}

class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        epoch_data = {
            "epoch": epoch + 1,
            "accuracy": float(logs.get("accuracy", 0)),
            "loss": float(logs.get("loss", 0)),
            "val_accuracy": float(logs.get("val_accuracy", 0)),
            "val_loss": float(logs.get("val_loss", 0))
        }

        training_status["epoch"] = epoch + 1
        training_status["accuracy"] = epoch_data["accuracy"]
        training_status["loss"] = epoch_data["loss"]
        training_status["val_accuracy"] = epoch_data["val_accuracy"]
        training_status["val_loss"] = epoch_data["val_loss"]

        training_status["epoch_logs"].append(epoch_data)
        training_status["message"] = f"Training Epoch {epoch+1}/{training_status['total_epochs']}..."


def train_meat_model(epochs=20):
    global training_status

    if not os.path.exists(DATASET_PATH):
        raise Exception("Dataset folder not found!")

    training_status["running"] = True
    training_status["done"] = False
    training_status["error"] = ""
    training_status["epoch"] = 0
    training_status["total_epochs"] = epochs
    training_status["epoch_logs"] = []
    training_status["message"] = "Preparing dataset..."

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(128, 128),
        batch_size=16,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(128, 128),
        batch_size=16,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    training_status["message"] = "Building model..."

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(3, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    training_status["message"] = "Training started..."

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[ProgressCallback()]
    )

    training_status["message"] = "Saving model..."
    model.save(MODEL_SAVE_PATH)

    # FINAL METRICS
    accuracy = float(history.history['accuracy'][-1])
    loss = float(history.history['loss'][-1])
    val_accuracy = float(history.history['val_accuracy'][-1])
    val_loss = float(history.history['val_loss'][-1])

    # CONFUSION MATRIX
    training_status["message"] = "Generating confusion matrix..."

    y_true = val_data.classes
    y_pred_probs = model.predict(val_data)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    labels = list(val_data.class_indices.keys())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")

    os.makedirs("../website/uploads", exist_ok=True)
    plt.title("Confusion Matrix - Meat Freshness Model")
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    training_status["confusion_matrix_image"] = "uploads/confusion_matrix.png"

    # SAVE REPORT JSON
    report_data = {
        "final_accuracy": round(accuracy, 4),
        "final_loss": round(loss, 4),
        "final_val_accuracy": round(val_accuracy, 4),
        "final_val_loss": round(val_loss, 4),
        "epoch_logs": training_status["epoch_logs"],
        "confusion_matrix_image": training_status["confusion_matrix_image"]
    }

    with open(REPORT_SAVE_PATH, "w") as f:
        json.dump(report_data, f)

    training_status["running"] = False
    training_status["done"] = True
    training_status["message"] = "Training completed!"

    return {
        "status": "success",
        "accuracy": round(accuracy, 4),
        "loss": round(loss, 4),
        "val_accuracy": round(val_accuracy, 4),
        "val_loss": round(val_loss, 4),
        "epoch_logs": training_status["epoch_logs"],
        "confusion_matrix_image": training_status["confusion_matrix_image"]
    }