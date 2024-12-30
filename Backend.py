pip install Flask tensorflow pandas scikit-learn # type: ignore

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Inisialisasi Flask
app = Flask(__name__)

# Memuat model .h5 yang sudah dilatih
model = load_model('D:\Document\School\Churn\Telco Churn\Coding\CNN+BiLSTM+SMOTEENN.h5')

# Inisialisasi LabelEncoder dan MinMaxScaler
le = LabelEncoder()  # Encoder untuk kategori
scaler = MinMaxScaler()  # Scaler untuk data numerik

# Fungsi untuk memproses inputan pengguna
def preprocess_input(data):
    # Encode fitur kategori
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    for col in categorical_columns:
        data[col] = le.fit_transform(data[col].astype(str))

    # Scale data numerik
    numeric_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data

# Endpoint untuk melakukan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Menerima data dari user
    data = request.get_json()  # Mengambil data JSON yang dikirim oleh frontend

    # Mengubah data menjadi DataFrame
    input_data = pd.DataFrame([data])

    # Memproses data input sebelum diberi ke model
    processed_data = preprocess_input(input_data)

    # Melakukan prediksi dengan model
    prediction = model.predict(processed_data)

    # Menentukan hasil prediksi
    result = 'Churn' if prediction[0] > 0.5 else 'No Churn'

    # Mengembalikan hasil prediksi sebagai JSON
    return jsonify({'prediction': result})

# Menjalankan server Flask
if __name__ == '__main__':
    app.run(debug=True)