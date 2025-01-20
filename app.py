import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

# Memuat model yang telah dilatih
# model_path = '/path/to/your/model/model_anemia_gender_v2.h5'  # Ganti dengan path model Anda di VPS
# model = load_model(model_path)

# Membuat instance aplikasi Flask
app = Flask(__name__)

# Konfigurasi untuk upload file
# app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder untuk menyimpan gambar sementara
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# # Membuat folder jika belum ada
# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# # Fungsi untuk memeriksa ekstensi file
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # Fungsi untuk memprediksi Hgb dan usia
# def predict_anemia(image_path, gender):
#     img = load_img(image_path, target_size=(224, 224))  # Memuat dan mengubah ukuran gambar
#     img_array = img_to_array(img) / 255.0  # Normalisasi gambar
#     img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch

#     gender_array = np.array([gender], dtype=np.float32).reshape(-1, 1)

#     # Pastikan model menerima dua input: gambar dan gender
#     prediction = model.predict([img_array, gender_array])

#     # Convert the results to native Python types for JSON serialization
#     hgb, age = prediction[0]
#     result = {"Hgb": float(hgb), "Age": float(age)}  # Convert to float
#     return result

# # Endpoint untuk prediksi
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image part in the request"}), 400

#     image = request.files['image']
#     gender = request.form.get('gender')

#     if image.filename == '' or gender is None:
#         return jsonify({"error": "Missing 'image' or 'gender' in request"}), 400

#     if not allowed_file(image.filename):
#         return jsonify({"error": "File type not allowed"}), 400

#     # Simpan gambar sementara
#     filename = secure_filename(image.filename)
#     image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     image.save(image_path)

#     try:
#         # Memanggil fungsi prediksi
#         result = predict_anemia(image_path, float(gender))
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# Endpoint untuk mengetes apakah aplikasi berjalan
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Aplikasi berjalan dengan baik!"}), 200

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
