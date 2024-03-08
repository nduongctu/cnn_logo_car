from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO


app = Flask(__name__)
model = load_model('C:/Users/Duong/PycharmProjects/CNN/best_logo_classification_model.h5')
class_names = ['hyundai', 'lexus', 'mazda', 'mercedes', 'opel', 'skoda', 'toyota', 'volkswagen']

@app.route('/')
def index():
    return render_template('/demo.html')

@app.route('/classify_image', methods=['POST'])
def classify_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        img_file = request.files['image']

        if img_file.filename == '':
            return jsonify({'error': 'No image selected'})

        # Đọc dữ liệu hình ảnh từ đối tượng FileStorage và chuyển đổi thành io.BytesIO
        img_bytes = BytesIO()
        img_file.save(img_bytes)
        img_bytes.seek(0)

        # Đọc và tiền xử lý ảnh
        img = image.load_img(img_bytes, target_size=(100, 100))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Dự đoán lớp của ảnh
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]

        return jsonify({'class': predicted_class})
    else:
        return jsonify({'error': 'Method not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
