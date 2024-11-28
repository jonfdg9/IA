import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
import os
import joblib


def column_family(X):
    return X[:, [0]] + X[:, [1]]

def family_name(function_transformer, feature_names_in):
    return ["family"] 

def family_pipeline():
    return make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    FunctionTransformer(column_family, feature_names_out=family_name))

def encode_sex(X):
    X[:, 0] = np.where(X[:, 0] == 'male', 0, 1) 
    return X

def categorize_age(X):
    pd.cut(X[:,0], bins = [-1, 16, 32, 48, 64, np.inf] ,labels = [1, 2, 3, 4, 5])
    return X


try:

    # Solicitar el nombre del archivo CSV por consolatita
    base_path = "/home/iabd/Escritorio/IA/SAPA/Machine Learning/"
    csv_path = input("Ingrese el nombre del archivo CSV: ").strip()
    full_path = os.path.join(base_path, csv_path)

    # Verificar si el archivo CSV existe
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"El archivo de datos '{full_path}' no existe.")
    
    # Cargar los datos del Titanic
    titanic = pd.read_csv(full_path)
    
    # Verificar si el archivo del modelo existe
    model_path = "/home/iabd/Escritorio/IA/SAPA/Machine Learning/modeloTitanic.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo del modelo '{model_path}' no existe.")
    
    # Cargar el modelo
    final_model_reloaded = joblib.load(model_path)
    
    # Verificar que los datos no estén vacíos
    if titanic.empty:
        raise ValueError("El archivo de datos está vacío. Proporcione un archivo con datos válidos.")
    
    # Seleccionar datos para predicción (primeras 5 filas)
    new_data = titanic[:5]
    
    # Verificar que las columnas requeridas para el modelo estén presentes
    model_features = getattr(final_model_reloaded, "feature_names_in_", None)  # Esto funciona si el modelo tiene 'feature_names_in_'
    if model_features is not None and not all(feature in new_data.columns for feature in model_features):
        missing_features = [feature for feature in model_features if feature not in new_data.columns]
        raise ValueError(f"Faltan las siguientes columnas necesarias para el modelo: {missing_features}")
    
    # Realizar predicciones
    predictions = final_model_reloaded.predict(new_data)
    print("Predicciones realizadas con éxito:")
    print(predictions)

except FileNotFoundError as fnf_error:
    print(f"Error: {fnf_error}")
except ValueError as val_error:
    print(f"Error: {val_error}")
except Exception as e:
    print(f"Ha ocurrido un error inesperado: {e}")