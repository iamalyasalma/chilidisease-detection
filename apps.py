from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from datetime import datetime

app = Flask(__name__)

# Path model lokal (karena sudah di repo GitHub)
MODEL_URL = "https://drive.google.com/uc?id=13tPaNC0RtyDty1HPuaihcmHaGYUFzsqK"
MODEL_PATH = "model_densenet.h5"


# Load model saat aplikasi mulai
print("🧠 Memuat model DenseNet...")
try:
    model_densenet = load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model_densenet = None


# Folder upload
UPLOAD_FOLDER = 'static/upload_gambar'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['upload_gambar'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/classification", methods=['GET'])
def classification():
    return render_template("classification.html")


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
        return render_template("index.html", error="Tipe file tidak didukung.")

    file_path = os.path.join(app.config['upload_gambar'], "temp_image.png")
    file.save(file_path)

    img = Image.open(file_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model_densenet.predict(img_array)
    class_names = ['Daun Bercak', 'Daun Gemini', 'Daun Layu', 'Daun Sehat']
    class_description = {
        'Daun Bercak': 'Daun ini memiliki bercak akibat infeksi jamur atau bakteri.',
        'Daun Gemini': 'Daun ini menunjukkan gejala virus Gemini yang menyebabkan warna kuning.',
        'Daun Layu': 'Daun ini mengalami kelayuan akibat infeksi bakteri atau kekurangan air.',
        'Daun Sehat': 'Daun ini sehat tanpa tanda-tanda penyakit.'
    }

    prediction_class = class_names[np.argmax(predictions)]
    confidence = f"{np.max(predictions) * 100:.2f}%"
    description = class_description.get(prediction_class, "Deskripsi tidak tersedia")

    now = datetime.now()
    result_image = f"static/upload_gambar/{now.strftime('%d%m%y-%H%M%S')}.png"
    img.save(result_image)

    return render_template(
        "index.html",
        img_path=result_image,
        prediction_densenet=prediction_class,
        confidence_densenet=confidence,
        description_densenet=description
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
