"""
Aplikasi Web Flask - Klasifikasi Jenis Hewan menggunakan Transfer Learning VGG16
"""

import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except FileExistsError:
    pass
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "model_animals10_vgg16_final.h5"
IMG_SIZE = (128, 128)   # HARUS SAMA dengan ukuran yang dipakai saat training!

# Urutan kelas HARUS SAMA dengan urutan folder saat training
# (flow_from_directory otomatis mengurutkan folder secara alfabetis)
class_names = [
    "Anjing",       # cane
    "Kuda",         # cavallo
    "Gajah",        # elefante
    "Kupu-kupu",    # farfalla
    "Ayam",         # gallina
    "Kucing",       # gatto
    "Sapi",         # mucca
    "Domba",        # pecora
    "Laba-laba",    # ragno
    "Tupai"         # scoiattolo
]

# ============================================================
# LOAD MODEL (sekali saja saat aplikasi start)
# ============================================================
print("Memuat model...")
model = load_model(MODEL_PATH)
print("Model berhasil dimuat.")


def predict_image(img_path):
    """Melakukan prediksi jenis hewan dari sebuah file gambar."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    predicted_label = class_names[predicted_index]
    return predicted_label, confidence


# ============================================================
# ROUTES
# ============================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    confidence_score = None
    image_path = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Pengaman ekstra: pastikan folder benar-benar ada tepat sebelum menyimpan
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
            except FileExistsError:
                pass

            file.save(filepath)

            predicted_label, confidence = predict_image(filepath)

            prediction_result = predicted_label
            confidence_score = round(confidence, 2)
            image_path = f"static/uploads/{filename}"   # path relatif untuk ditampilkan di HTML

    return render_template(
        'index.html',
        prediction=prediction_result,
        confidence=confidence_score,
        image_path=image_path
    )


if __name__ == '__main__':
    app.run(debug=True)
