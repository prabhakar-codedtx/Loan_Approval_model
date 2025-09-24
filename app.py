from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import requests

app = Flask(__name__)

# === Google Drive File IDs ===
GDRIVE_CLASS_MODEL_ID = "1qF1kB9dNIW-QWmr5td27OoQ2rQYpw0JK"
GDRIVE_REG_MODEL_ID = "1pqXWkgx38oannunWu8ka5t5jZxJysiaI"

# === Download Function ===
def download_from_google_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {dest_path} from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ {dest_path} downloaded successfully.")
        else:
            print(f"❌ Failed to download {dest_path} (Status code: {response.status_code})")

# === Download Models if Missing ===
download_from_google_drive(GDRIVE_CLASS_MODEL_ID, "model_clas.pkl")
download_from_google_drive(GDRIVE_REG_MODEL_ID, "model_reg.pkl")

# === Load Models ===
try:
    with open("model_clas.pkl", "rb") as f:
        model_clas = pickle.load(f)

    with open("model_reg.pkl", "rb") as f:
        model_reg = pickle.load(f)
except Exception as e:
    print(f"❌ Failed to load model files: {str(e)}")
    raise

# === Feature Columns ===
FEATURES_CLAS = [
    'no_of_dependents','education','self_employed','income_annum',
    'loan_amount','loan_term','cibil_score',
    'residential_assets_value','commercial_assets_value'
]

FEATURES_REG = [
    'no_of_dependents','education','self_employed','income_annum',
    'loan_amount','loan_term','residential_assets_value','commercial_assets_value','loan_status'
]

# === Routes ===

@app.route('/', methods=['GET'])
def ping():
    return jsonify({"message": "✅ Server is running"}), 200

@app.route('/predict_clas', methods=['POST'])
def predict_clas():
    data = request.json

    # Validate input
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Invalid input format"}), 400

    # Filter features
    try:
        df = df[FEATURES_CLAS]
    except KeyError as e:
        return jsonify({"error": f"Missing feature(s): {str(e)}"}), 400

    # Make prediction
    pred = model_clas.predict(df)
    return jsonify({"prediction": pred.tolist()}), 200

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
    return jsonify({"prediction": pred.tolist()}), 200

# === Run Server (for local testing) ===
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
