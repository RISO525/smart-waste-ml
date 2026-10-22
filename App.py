
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path
import os

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_output"

fill_model = joblib.load(
    MODEL_DIR / "fill_prediction_model.joblib"
)
alert_model = joblib.load(
    MODEL_DIR / "collection_alert_model.joblib"
)

numeric_features = [
    "hour", "day_of_week", "is_weekend", "is_busy_period",
    "fill_percent", "previous_fill_percent", "fill_change_30m",
    "minutes_since_collection"
]

categorical_features = ["bin_id", "location"]
features = numeric_features + categorical_features


@app.route("/")
def home():
    return "Smart Waste Management ML API is running!"


@app.route("/health")
def health():
    return jsonify({"status": "running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    missing = [f for f in features if f not in data]
    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing": missing
        }), 400

    try:
        row = pd.DataFrame(
            [{f: data[f] for f in features}],
            columns=features
        )

        for f in numeric_features:
            row[f] = pd.to_numeric(row[f], errors="raise")

        predicted_fill = float(fill_model.predict(row)[0])
        predicted_fill = round(max(0, min(100, predicted_fill)), 1)

        alert = int(alert_model.predict(row)[0])

        probability = None
        if hasattr(alert_model, "predict_proba"):
            classes = list(alert_model.classes_)
            if 1 in classes:
                probability = round(
                    float(alert_model.predict_proba(row)[0][classes.index(1)]),
                    3
                )

        return jsonify({
            "bin_id": str(data["bin_id"]),
            "predicted_fill_percent_60m": predicted_fill,
            "collection_alert": bool(alert),
            "alert_probability": probability
        })

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
