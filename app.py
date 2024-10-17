from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

# Load the deep learning model and scaler
try:
    deep_model = load_model('models/heart_attack_model.h5')
except FileNotFoundError:
    raise RuntimeError("Model file not found. Please check the path to 'models/heart_attack_model.h5'.")
    
try:
    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Scaler file not found. Please check the path to 'data/scaler.pkl'.")

# Hardcoded metrics (should be dynamically loaded in a real scenario)
metrics = {
    'Accuracy': 0.85,
    'Precision': 0.87,
    'Recall': 0.83,
    'F1 Score': 0.85,
    'ROC AUC Score': 0.90
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        features = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['family_history']),
            float(request.form['smoking']),
            float(request.form['alcohol']),
            float(request.form['physical_activity']),
            float(request.form['diet']),
            float(request.form['bmi']),
            float(request.form['blood_pressure']),
            float(request.form['cholesterol']),
            float(request.form['diabetes']),
            float(request.form['stress'])
        ]
        
        # Scale input data
        input_data = np.array([features])
        input_data = scaler.transform(input_data)
        
        # Predict
        prediction = deep_model.predict(input_data)
        risk = prediction[0][0]
        
        return render_template('result.html', risk=risk * 100, metrics=metrics)
        
if __name__ == '__main__':
    app.run(debug=True)

