from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model('emotion_model.keras')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'model': 'Emotion Recognition',
        'accuracy': '67.90%',
        'usage': 'POST /predict with image file'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('L')
        image = image.resize((48, 48))
        image_array = np.array(image, dtype='float32').reshape(1, 48, 48, 1) / 255.0
        prediction = model.predict(image_array, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = emotions[emotion_idx]
        confidence = float(np.max(prediction))
        all_preds = {emotions[i]: float(prediction[0][i]) for i in range(7)}
        return jsonify({'emotion': emotion, 'confidence': f'{confidence:.2%}', 'all_predictions': all_preds})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

