
from flask import Flask, render_template, request, jsonify
from keras.models import load_model

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.densenet import DenseNet121

import tensorflow as tf
from tensorflow import keras
from skimage import transform, io
import numpy as np
import os
import requests
from PIL import Image
from datetime import datetime
from keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)


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

# Load model setelah file tersedia
model_densenet = load_model(model_path)

# Load model DenseNet 
# model_densenet = load_model("model_densenet.h5") 

# Membuat tempat upload gambar
upload_folder = 'static/upload_gambar'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)  # Buat direktori jika belum ada
app.config['upload_gambar'] = upload_folder
allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'} #extension yang diperbolehkan 


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions




# Tampilan prediksi
@app.route('/', methods=['GET', 'POST'])
def prediksi():
    return render_template("index.html")

@app.route("/classification", methods = ['GET', 'POST'])
def classification():
	return render_template("classification.html")

@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", error="No file part in the request.")
    files = request.files.getlist('file')
    filename = "temp_image.png"
    errors = {}
    success = False

    for file in files:
        if file.filename == '':
            return render_template("index.html", error=f"No file selected for uploading")
    
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


    # Preprocessing gamabar
    image_pil = Image.open(img_url).resize((224, 224)) #ukuran gambar untuk model DenseNet
    now = datetime.now()
    predict_image_path = 'static/upload_gambar/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    image_pil.save(predict_image_path) #simpan gambar

    #convert gambar menjadi array numpy dan normalisasi
    image_array = np.array(image_pil) / 255.0 #normalisasi nilai piksel 
    image_array= np.expand_dims(image_array, axis=0)

    # return image_array, predict_image_path

    class_description = {
        'Daun Bercak' : 'Daun ini memiliki bercak-bercak yang disebabkan oleh infeksi jamur atau bakteri.',
        'Daun Gemini' : 'Daun ini menunjukkan gejala virus Gemini yang menyebabkan warna kuning pada daun.',
        'Daun Layu' : 'Daun ini mengalami kelayuan yang biasanya disebabkan oleh infeksi bakteri atau kekurangan air.',
        'Daun Sehat' : 'Daun ini sehat tanpa tanda-tanda penyakit atau kerusakan.',
    }
    
    #Prediksi
    prediction_array_densenet = model_densenet.predict(image_array)
    class_names = ['Daun Bercak', 'Daun Gemini', 'Daun Layu', 'Daun Sehat']  

    # Define class names
    prediction_class = class_names[np.argmax(prediction_array_densenet)]
    confidence = '{:.2f}%'.format(100 * np.max(prediction_array_densenet))
    #Ambil deskripsi dari dictionary
    description = class_description.get(prediction_class, 'Deskripsi tidak tersedia')

    return render_template(
        "index.html", 
        img_path = predict_image_path, #path gambar untuk ditampilkan
        prediction_densenet = prediction_class,
        confidence_densenet = confidence,
        description_densenet= description
        
                           )

if __name__ == '__main__':
    app.run(debug=True)
    
