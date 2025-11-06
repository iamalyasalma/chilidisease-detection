from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import os
import requests
from PIL import Image
from datetime import datetime
from flask_cors import CORS

# === Inisialisasi Flask ===
app = Flask(__name__)
CORS(app)

# === Konfigurasi Path Model ===
# Ganti link di bawah dengan link model kamu di Hugging Face
MODEL_URL = "https://huggingface.co/halloalya/chili-leaf-disease-model/resolve/a901ebdc537c835f3321cccffb3e14323a76c6ce/model_densenet.h5"
MODEL_PATH = "model_densenet.h5"

# === Unduh Model Jika Belum Ada ===
if not os.path.exists(MODEL_PATH):
    print("📥 Mengunduh model dari Hugging Face...")
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model berhasil diunduh!")
        print("📦 Ukuran file:", os.path.getsize(MODEL_PATH), "bytes")
    except Exception as e:
        print(f"❌ Gagal mengunduh model: {e}")

# === Load Model ===
print("🧠 Memuat model DenseNet...")
try:
    model_densenet = load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model_densenet = None

# === Folder Upload ===
UPLOAD_FOLDER = 'static/upload_gambar'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# === Route Halaman Utama ===
@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


# === Route Halaman Klasifikasi ===
@app.route("/classification", methods=['GET'])
def classification():
    return render_template("classification.html")


# === Route Prediksi ===
@app.route('/submit', methods=['POST'])
def predict():
    if model_densenet is None:
        return render_template("index.html", error="Model belum siap. Coba refresh halaman.")

    if 'file' not in request.files:
        return render_template("index.html", error="Tidak ada file yang diunggah.")

    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", error="Tidak ada file yang dipilih.")

    if not allowed_file(file.filename):
        return render_template("index.html", error="Tipe file tidak didukung. Harap unggah JPG/PNG.")

    # Simpan file upload
    now = datetime.now()
    filename = now.strftime("%d%m%y-%H%M%S") + ".png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocessing gambar
    img = Image.open(filepath).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model_densenet.predict(img_array)
    class_names = ['Daun Bercak', 'Daun Gemini', 'Daun Layu', 'Daun Sehat']
    prediction_class = class_names[np.argmax(predictions)]
    confidence = f"{np.max(predictions) * 100:.2f}%"

    class_description = {
        'Daun Bercak': 'Daun ini memiliki bercak akibat infeksi jamur atau bakteri.',
        'Daun Gemini': 'Daun ini menunjukkan gejala virus Gemini yang menyebabkan warna kuning.',
        'Daun Layu': 'Daun ini mengalami kelayuan akibat infeksi bakteri atau kekurangan air.',
        'Daun Sehat': 'Daun ini sehat tanpa tanda-tanda penyakit.'
    }

    description = class_description.get(prediction_class, "Deskripsi tidak tersedia.")

    return render_template(
        "index.html",
        img_path=filepath,
        prediction_densenet=prediction_class,
        confidence_densenet=confidence,
        description_densenet=description
    )


# === Jalankan Flask ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
