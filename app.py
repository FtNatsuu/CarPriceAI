from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# Créer l'application Flask
app = Flask(__name__)

# Charger le modèle et le préprocesseur
model = load_model('model.keras')  # Modèle Keras
preprocessor = joblib.load('preprocessor.pkl')  # Préprocesseur scikit-learn

# Liste des options pour les champs catégoriels
brands = ["Kia", "Chevrolet", "Mercedes", "Audi", "Volkswagen", "Toyota", "Honda", "BMW", "Hyundai", "Ford"]
models = ["Rio", "Malibu", "GLA", "Q5", "Golf", "Camry", "Civic", "Sportage", "RAV4", "5 Series", "CR-V", "Elantra",
          "Tiguan", "Equinox", "Explorer", "A3", "3 Series", "Tucson", "Passat", "Impala", "Corolla", "Optima", "Fiesta",
          "A4", "Focus", "E-Class", "Sonata", "C-Class", "X5", "Accord"]
transmissions = ["Manual", "Automatic", "Semi-Automatic"]
fuel_types = ["Diesel", "Hybrid", "Electric", "Petrol"]

# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template('index.html', brands=brands, models=models, transmissions=transmissions, fuel_types=fuel_types)

# Route pour traiter les données soumises et faire la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données de l'utilisateur
        brand = request.form['brand']
        model_name = request.form['model']
        transmission = request.form['transmission']
        fuel_type = request.form['fuel_type']
        year = int(request.form['year'])
        engine_size = float(request.form['engine_size'])
        mileage = float(request.form['mileage'])
        doors = int(request.form['doors'])
        owner_count = int(request.form['owner_count'])

        # Créer un DataFrame avec les données de l'utilisateur
        user_data = pd.DataFrame([[brand, model_name, transmission, fuel_type, year, engine_size, mileage, doors, owner_count]],
                                 columns=["Brand", "Model", "Transmission", "Fuel_Type", "Year", "Engine_Size", "Mileage", "Doors", "Owner_Count"])

        # Appliquer le préprocesseur
        processed_data = preprocessor.transform(user_data)

        # Faire la prédiction
        prediction = model.predict(processed_data)
        
        # Afficher le résultat de la prédiction
        predicted_price = round(prediction[0][0], 2)
        return render_template('result.html', price=predicted_price)

    except Exception as e:
        return f"An error occurred: {e}"

# Lancer l'application Flask
if __name__ == "__main__":
    app.run(debug=True)
