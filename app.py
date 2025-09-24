from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import requests

app = Flask(__name__)

# === Google Drive File IDs ===
GDRIVE_CLASS_MODEL_ID = "1qF1kB9dNIW-QWmr5td27OoQ2rQYpw0JK"
GDRIVE_REG_MODEL_ID = "1pqXWkgx38oannunWu8ka5t5jZxJysiaI"

# === Google Drive Download Handler ===
def download_from_google_drive(file_id, dest_path):
    if os.path.exists(dest_path):
        print(f"✅ {dest_path} already exists, skipping download.")
        return

    print(f"📥 Downloading {dest_path} from Google Drive...")

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, dest_path)
    print(f"✅ {dest_path} downloaded successfully. Size: {os.path.getsize(dest_path)} bytes")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# === Download Models ===
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

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Invalid input format. Send a JSON object or list."}), 400

    try:
        df = df[FEATURES_CLAS]
    except KeyError as e:
        return jsonify({"error": f"Missing feature(s): {str(e)}"}), 400

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
        return jsonify({"error": "Invalid input format. Send a JSON object or list."}), 400

    try:
        df = df[FEATURES_REG]
    except KeyError as e:
        return jsonify({"error": f"Missing feature(s): {str(e)}"}), 400

    pred = model_reg.predict(df)
    return jsonify({"prediction": pred.tolist()}), 200

# === Run Server Locally ===
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
