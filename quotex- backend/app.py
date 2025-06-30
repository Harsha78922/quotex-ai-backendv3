from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("quotex_signal_model.h5")

@app.route("/", methods=["GET"])
def home():
    return "Quotex Signal Backend Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        signal = "Buy" if prediction[0][0] > 0.5 else "Sell"
        return jsonify({"signal": signal})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
