from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Flask Uygulaması
app = Flask(__name__)

# Kaydedilen model ve etiket kodlayıcıyı yükleme
model = joblib.load('best_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
symptoms = joblib.load('symptoms.pkl')

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input_data = [int(data[symptom]) for symptom in symptoms]

    # Girdi verilerini modele uygun formata dönüştürme
    input_data = np.array([input_data])
    probabilities = model.predict_proba(input_data)[0]
    
    # Tahmin edilen hastalıkları, olasılıklarıyla birlikte almak
    disease_probabilities = {
        label_encoder.inverse_transform([i])[0]: f"{prob * 100:.2f}%"
        for i, prob in enumerate(probabilities)
    }

    # Sonuçları JSON formatında döndürme
    return jsonify(disease_probabilities)

if __name__ == "__main__":
    app.run(debug=True)






