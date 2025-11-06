from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
from keras.applications.densenet import DenseNet121
from PIL import Image
import numpy as np
import os
import gdown
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# URL ke file model Anda (pastikan ini versi uc?id=...)
MODEL_URL = "https://drive.google.com/uc?id=13tPaNC0RtyDty1HPuaihcmHaGYUFzsqK"
MODEL_PATH = "model_densenet.h5"
model_densenet = None  # model belum dimuat saat startup

# Folder upload
UPLOAD_FOLDER = 'static/upload_gambar'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['upload_gambar'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_if_needed():
    """Lazy load model hanya saat pertama kali digunakan"""
    global model_densenet
    if model_densenet is None:
        if not os.path.exists(MODEL_PATH):
            print("📥 Downloading model from Google Drive...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            print("✅ Model downloaded.")

        print("🧠 Loading model into memory...")
        try:
            model_densenet = load_model(MODEL_PATH)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e


@app.route('/', methods=['GET', 'POST'])
def prediksi():
    return render_template("index.html")


@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classification.html")


@app.route('/submit', methods=['POST'])
def predict():
    load_model_if_needed()  # pastikan model siap

    if 'file' not in request.files:
        return render_template("index.html", error="No file part in the request.")

    files = request.files.getlist('file')
    filename = "temp_image.png"
    errors = {}
    success = False

    for file in files:
        if file.filename == '':
            return render_template("index.html", error="No file selected for uploading")

    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['upload_gambar'], filename))
            success = True
        if not allowed_file(file.filename):
            return render_template("index.html", error=f"File type not allowed: {file.filename}")

    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp

    img_url = os.path.join(app.config['upload_gambar'], filename)

    # Preprocessing gambar
    image_pil = Image.open(img_url).resize((224, 224))
    now = datetime.now()
    predict_image_path = os.path.join('static/upload_gambar', now.strftime("%d%m%y-%H%M%S") + ".png")
    image_pil.save(predict_image_path)

    image_array = np.array(image_pil) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Klasifikasi
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
    description = class_description.get(prediction_class, 'Deskripsi tidak tersedia')

    return render_template(
        "index.html",
        img_path=predict_image_path,
        prediction_densenet=prediction_class,
        confidence_densenet=confidence,
        description_densenet=description
    )


if __name__ == '__main__':
    # host dan port biar cocok di Railway
    app.run(host='0.0.0.0', port=8080)
