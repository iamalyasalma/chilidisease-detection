from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import os
from datetime import datetime
from PIL import Image

app = Flask(__name__)

# Konfigurasi folder upload
app.config['upload_folder'] = 'static/upload_folder/'
allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

# URL ke file model Anda
model_url = "https://drive.google.com/uc?id=1fO4FQKV6XvgjzFz4BeNXjJZZWv4bQszj"
model_path = "model_densenet.h5"

# Unduh model jika belum ada
if not os.path.exists(model_path):
    print("Downloading model file...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded.")

# Load model
model_densenet = load_model(model_path)

# Fungsi validasi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Route utama
@app.route('/', methods=['GET', 'POST'])
def prediksi():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def predict():
    # Validasi file
    if 'file' not in request.files:
        return render_template("index.html", error="No file part in the request.")
    
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return render_template("index.html", error="No file selected for uploading")
    
    # Pastikan folder upload tersedia
    if not os.path.exists(app.config['upload_folder']):
        os.makedirs(app.config['upload_folder'])
    
    # Simpan file dengan nama unik
    now = datetime.now()
    filename = now.strftime("%d%m%y-%H%M%S") + "_uploaded.png"
    file_path = os.path.join(app.config['upload_folder'], filename)
    files[0].save(file_path)
    
    # Preprocessing gambar
    image_pil = Image.open(file_path).resize((224, 224))
    image_array = np.array(image_pil) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Deskripsi kelas
    class_description = {
        'Daun Bercak': 'Daun ini memiliki bercak-bercak yang disebabkan oleh infeksi jamur atau bakteri.',
        'Daun Gemini': 'Daun ini menunjukkan gejala virus Gemini yang menyebabkan warna kuning pada daun.',
        'Daun Layu': 'Daun ini mengalami kelayuan yang biasanya disebabkan oleh infeksi bakteri atau kekurangan air.',
        'Daun Sehat': 'Daun ini sehat tanpa tanda-tanda penyakit atau kerusakan.',
    }
    
    # Prediksi menggunakan model
    predictions = model_densenet.predict(image_array)
    class_names = list(class_description.keys())
    predicted_class = class_names[np.argmax(predictions)]
    confidence = f"{100 * np.max(predictions):.2f}%"
    description = class_description.get(predicted_class, "Deskripsi tidak tersedia.")
    
    # Kirim hasil ke template
    return render_template(
        "index.html",
        img_path=file_path,
        prediction_densenet=predicted_class,
        confidence_densenet=confidence,
        description_densenet=description
    )

if __name__ == '__main__':
    app.run(debug=True)
