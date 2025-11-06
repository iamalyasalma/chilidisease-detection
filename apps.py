from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
import tensorflow as tf
from skimage import transform, io
import numpy as np
import os
import requests
from PIL import Image
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# URL ke file model Anda
model_url = "https://drive.google.com/uc?export=download&id=13tPaNC0RtyDty1HPuaihcmHaGYUFzsqK"
model_path = "model_densenet.h5"

# --- Cek dan download model hanya kalau belum ada ---
def download_model():
    if not os.path.exists(model_path):
        print("⬇️ Mengunduh model dari Google Drive...")
        response = requests.get(model_url, allow_redirects=True)
        if response.status_code == 200:
            content = response.content

            # ✅ Pastikan file valid (bukan halaman HTML error)
            if b'HDF' not in content[:100]:
                with open("download_error.html", "wb") as f:
                    f.write(content)
                raise ValueError("⚠️ File yang diunduh bukan file .h5 valid. Cek download_error.html!")

            with open(model_path, "wb") as f:
                f.write(content)
            print("✅ Model berhasil diunduh!")
        else:
            raise ValueError(f"⚠️ Gagal download model (status code {response.status_code})")


# --- Load model dengan aman ---
def load_densenet():
    global model_densenet
    try:
        if not os.path.exists(model_path):
            download_model()
        print("🧠 Memuat model...")
        model_densenet = load_model(model_path)
        print("✅ Model berhasil dimuat!")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        model_densenet = None


# Panggil saat startup
model_densenet = None
load_densenet()

# Membuat tempat upload gambar
upload_folder = 'static/upload_gambar'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
app.config['upload_gambar'] = upload_folder
allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# Tampilan utama
@app.route('/', methods=['GET', 'POST'])
def prediksi():
    return render_template("index.html")


@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classification.html")


@app.route('/submit', methods=['POST'])
def predict():
    try:
        if model_densenet is None:
            return render_template("index.html", error="Model belum siap. Coba refresh halaman.")

        if 'file' not in request.files:
            return render_template("index.html", error="Tidak ada file yang dikirim.")

        files = request.files.getlist('file')
        filename = "temp_image.png"
        success = False

        for file in files:
            if file.filename == '':
                return render_template("index.html", error="Tidak ada file yang dipilih.")

        for file in files:
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config['upload_gambar'], filename))
                success = True
            else:
                return render_template("index.html", error=f"File tidak diizinkan: {file.filename}")

        if not success:
            return render_template("index.html", error="Gagal upload file.")

        img_url = os.path.join(app.config['upload_gambar'], filename)

        # --- Preprocessing gambar ---
        image_pil = Image.open(img_url).resize((224, 224))
        now = datetime.now()
        predict_image_path = f'static/upload_gambar/{now.strftime("%d%m%y-%H%M%S")}.png'
        image_pil.save(predict_image_path)

        image_array = np.array(image_pil) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # --- Prediksi ---
        class_description = {
            'Daun Bercak': 'Daun ini memiliki bercak-bercak yang disebabkan oleh infeksi jamur atau bakteri.',
            'Daun Gemini': 'Daun ini menunjukkan gejala virus Gemini yang menyebabkan warna kuning pada daun.',
            'Daun Layu': 'Daun ini mengalami kelayuan yang biasanya disebabkan oleh infeksi bakteri atau kekurangan air.',
            'Daun Sehat': 'Daun ini sehat tanpa tanda-tanda penyakit atau kerusakan.',
        }

        prediction_array = model_densenet.predict(image_array)
        class_names = ['Daun Bercak', 'Daun Gemini', 'Daun Layu', 'Daun Sehat']

        prediction_class = class_names[np.argmax(prediction_array)]
        confidence = '{:.2f}%'.format(100 * np.max(prediction_array))
        description = class_description.get(prediction_class, 'Deskripsi tidak tersedia.')

        return render_template(
            "index.html",
            img_path=predict_image_path,
            prediction_densenet=prediction_class,
            confidence_densenet=confidence,
            description_densenet=description
        )

    except Exception as e:
        print(f"❌ ERROR di /submit: {e}")
        return render_template("index.html", error=f"Terjadi kesalahan: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
