from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
from keras.applications.densenet import DenseNet121
import tensorflow as tf
from tensorflow import keras
from skimage import transform, io
import numpy as np
import os
import gdown  # ✅ tambahan untuk download dari Google Drive
from PIL import Image
from datetime import datetime
from keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)

# URL ke file model di Google Drive
model_url = "https://drive.google.com/uc?id=1fO4FQKV6XvgjzFz4BeNXjJZZWv4bQszj"
model_path = "model_densenet.h5"

# 🔽 Unduh model dari Google Drive jika belum ada
if not os.path.exists(model_path):
    print("📥 Mengunduh model dari Google Drive...")
    try:
        gdown.download(model_url, model_path, quiet=False, fuzzy=True)
        print("✅ Model berhasil diunduh!")
        print("📦 Ukuran file:", os.path.getsize(model_path), "bytes")
    except Exception as e:
        print(f"❌ Gagal mengunduh model: {e}")

# 🧠 Load model setelah file tersedia
try:
    model_densenet = load_model(model_path)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model_densenet = None

# Membuat tempat upload gambar
upload_folder = 'static/upload_gambar'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
app.config['upload_gambar'] = upload_folder
allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Tampilan prediksi
@app.route('/', methods=['GET', 'POST'])
def prediksi():
    return render_template("index.html")

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classification.html")

@app.route('/submit', methods=['POST'])
def predict():
    if model_densenet is None:
        return render_template("index.html", error="Model belum siap. Coba refresh halaman.")

    if 'file' not in request.files:
        return render_template("index.html", error="Tidak ada file yang diunggah.")
    
    files = request.files.getlist('file')
    filename = "temp_image.png"
    errors = {}
    success = False

    for file in files:
        if file.filename == '':
            return render_template("index.html", error="Tidak ada file yang dipilih.")
    
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['upload_gambar'], filename))
            success = True
        else:
            return render_template("index.html", error=f"Tipe file tidak didukung: {file.filename}")
        
    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp
    
    img_url = os.path.join(app.config['upload_gambar'], filename)

    # Preprocessing gambar
    image_pil = Image.open(img_url).resize((224, 224))
    now = datetime.now()
    predict_image_path = 'static/upload_gambar/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    image_pil.save(predict_image_path)

    image_array = np.array(image_pil) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    class_description = {
        'Daun Bercak': 'Daun ini memiliki bercak-bercak yang disebabkan oleh infeksi jamur atau bakteri.',
        'Daun Gemini': 'Daun ini menunjukkan gejala virus Gemini yang menyebabkan warna kuning pada daun.',
        'Daun Layu': 'Daun ini mengalami kelayuan yang biasanya disebabkan oleh infeksi bakteri atau kekurangan air.',
        'Daun Sehat': 'Daun ini sehat tanpa tanda-tanda penyakit atau kerusakan.',
    }
    
    # Prediksi
    prediction_array_densenet = model_densenet.predict(image_array)
    class_names = ['Daun Bercak', 'Daun Gemini', 'Daun Layu', 'Daun Sehat']  

    prediction_class = class_names[np.argmax(prediction_array_densenet)]
    confidence = '{:.2f}%'.format(100 * np.max(prediction_array_densenet))
    description = class_description.get(prediction_class, 'Deskripsi tidak tersedia')

    return render_template(
        "index.html", 
        img_path=predict_image_path,
        prediction_densenet=prediction_class,
        confidence_densenet=confidence,
        description_densenet=description
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
