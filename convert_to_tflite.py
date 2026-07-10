import tensorflow as tf

KERAS_MODEL_PATH = "models/meat_model.keras"
TFLITE_MODEL_PATH = "models/meat_model.tflite"

model = tf.keras.models.load_model(KERAS_MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# OPTIONAL optimizations (recommended)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved to:", TFLITE_MODEL_PATH)