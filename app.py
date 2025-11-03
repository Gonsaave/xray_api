from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'pneumonia_model.hdf5'  # Cambia esto a la ruta de tu modelo
model = None

def load_model():
    """Carga el modelo al iniciar la aplicación"""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesa la imagen para el modelo
    """
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar
    image = image.resize(target_size)
    
    # Convertir a array numpy
    img_array = np.array(image)
    
    # Normalizar
    img_array = img_array / 150.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def home():
    """
    Endpoint inicial
    """
    return jsonify({
        'message': 'API de Predicción de Neumonía',
        'status': 'online',
        'endpoints': {
            '/predict': 'POST - Envía una imagen para predicción'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para predicción de pneumonía
    """
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    try:
        # Opción file request
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No se seleccionó archivo'}), 400
            
            image = Image.open(file.stream)
        
        # Opción base64
        elif 'image' in request.json:
            image_data = request.json['image']
            # Remover el prefijo data:image si existe
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        
        else:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        # Preprocesar imagen
        processed_image = preprocess_image(image)
        
        # Hacer predicción
        prediction = model.predict(processed_image, verbose=0)
        
        # Interpretar resultado
        probability = float(prediction[0][0])
        
        # Determinar clase
        if probability > 0.5:
            diagnosis = 'Neumonía'
            confidence = probability
        else:
            diagnosis = 'Normal'
            confidence = 1 - probability
        
        return jsonify({
            'diagnosis': diagnosis,
            'confidence': round(confidence * 100, 2),
            'probability_pneumonia': round(probability * 100, 2),
            'probability_normal': round((1 - probability) * 100, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint estado del servicio
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })
