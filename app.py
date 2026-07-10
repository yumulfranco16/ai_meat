from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from threading import Thread

from train_model import train_meat_model, training_status

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/meat_model.keras"
LABELS = ["fresh", "semi_fresh", "spoiled"]

training_thread = None
model = None

def load_model_once():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("Model file not found. Please train first.")

@app.route("/train_start", methods=["GET"])
def train_start():
    global training_thread

    if training_status["running"]:
        return jsonify({"status": "error", "message": "Training is already running."})

    def run_training():
        try:
            train_meat_model(epochs=10)
            tf.keras.backend.clear_session()
            load_model_once()
        except Exception as e:
            training_status["running"] = False
            training_status["done"] = True
            training_status["error"] = str(e)
            training_status["message"] = "Training failed!"

    training_thread = Thread(target=run_training)
    training_thread.start()

    return jsonify({"status": "success", "message": "Training started."})


@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(training_status)


@app.route("/predict", methods=["POST"])
def predict():
    global model

    if model is None:
        load_model_once()
        
    if model is None:
        return jsonify({"error": "Model not loaded. Train the model first."})

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']

    # create unique filename to prevent conflicts
    import time
    temp_path = os.path.join("temp_uploads", f"scan_{time.time()}.jpg")
    file.save(temp_path)

    try:
        img = load_img(temp_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]
        predicted_index = np.argmax(prediction)

        predicted_label = LABELS[predicted_index]
        confidence = float(prediction[predicted_index]) * 100

        # delete temp file after prediction
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "prediction": predicted_label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        # delete temp file even if error happens
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({"error": str(e)})

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("temp_uploads", exist_ok=True)

    # app.run(host="127.0.0.1", port=5000, debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)