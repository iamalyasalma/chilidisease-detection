from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import os

app = Flask(__name__)
CORS(app)

# --- Konfigurasi model ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=13tPaNC0RtyDty1HPuaihcmHaGYUFzsqK"
MODEL_PATH = "model_densenet.h5"
model_densenet = None


# --- Fungsi download model ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, allow_redirects=True)
        if response.status_code == 200:
            # Pastikan file bukan HTML
            if response.text.startswith("<!DOCTYPE html>"):
                raise ValueError("⚠️ Gagal download model — Google Drive mengirim halaman HTML, bukan file .h5")
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("✅ Model downloaded successfully!")
        else:
            raise ValueError(f"⚠️ Download failed with status code {response.status_code}")


# --- Fungsi load model ---
def load_model_if_needed():
    global model_densenet
    if model_densenet is None:
        download_model()
        print("🧠 Loading model into memory...")
        try:
            model_densenet = load_model(MODEL_PATH)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e


# --- Fungsi prediksi ---
@app.route('/submit', methods=['POST'])
def predict():
    try:
        load_model_if_needed()  # pastikan model siap

        # Ambil file gambar dari request
        file = request.files['file']
        file_path = "temp.jpg"
        file.save(file_path)

        # Preprocessing gambar
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Prediksi
        predictions = model_densenet.predict(img_array)
        class_names = ['Bercak Daun', 'Layu Bakteri Ralstonia', 'Sehat', 'Virus Kuning']
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Jalankan aplikasi ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
