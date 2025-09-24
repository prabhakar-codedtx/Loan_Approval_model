from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load models
with open("model_clas.pkl", "rb") as f:
    model_clas = pickle.load(f)

with open("model_reg.pkl", "rb") as f:
    model_reg = pickle.load(f)

# Features used during training
FEATURES_CLAS = [
    'no_of_dependents','education','self_employed','income_annum',
    'loan_amount','loan_term','cibil_score',
    'residential_assets_value','commercial_assets_value'
]

FEATURES_REG = [
    'no_of_dependents','education','self_employed','income_annum',
    'loan_amount','loan_term','residential_assets_value','commercial_assets_value','loan_status'
]

@app.route('/predict_clas', methods=['POST'])
def predict_clas():
    data = request.json

    # Convert JSON → DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Invalid input format"}), 400

    # Keep only training features
    try:
        df = df[FEATURES_CLAS]
    except KeyError as e:
        return jsonify({"error": f"Missing feature(s): {str(e)}"}), 400

    pred = model_clas.predict(df)
    return jsonify({"prediction": pred.tolist()})


@app.route('/predict_reg', methods=['POST'])
def predict_reg():
    data = request.json

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Invalid input format"}), 400

    try:
        df = df[FEATURES_REG]
    except KeyError as e:
        return jsonify({"error": f"Missing feature(s): {str(e)}"}), 400

    pred = model_reg.predict(df)
    return jsonify({"prediction": pred.tolist()})


@app.route('/', methods=['GET'])
def ping():
    return jsonify({"message": "✅ Server is running"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
