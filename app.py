
from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import re

app = Flask(__name__)
xgb_model = joblib.load("xgboost_model.pkl")
rnn_model = load_model("rnn_model.h5")

def preprocess(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    question = preprocess(data.get("question", ""))
    if not question:
        return jsonify({"error": "Invalid input"}), 400
    x_input = np.array([len(question.split())]).reshape(1, -1)
    rnn_input = np.array([[ord(c) % 256 for c in question[:100]]])
    xgb_pred = xgb_model.predict(x_input)
    rnn_pred = rnn_model.predict(rnn_input, verbose=0)
    final_response = f"XGBoost: {xgb_pred[0]:.2f}, RNN: {rnn_pred[0][0]:.2f} — Homework help is on the way!"
    return jsonify({"response": final_response})

if __name__ == "__main__":
    app.run(debug=True)
