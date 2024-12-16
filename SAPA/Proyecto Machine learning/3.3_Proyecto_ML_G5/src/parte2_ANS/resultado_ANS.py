import pandas as pd
import joblib
import os
from PIL import Image
import numpy as np

def main():
    """
    Carga un modelo previamente entrenado y realiza predicciones sobre una imagen proporcionada por el usuario.

    Este script realiza las siguientes acciones:
    1. Solicita al usuario la ruta de una imagen.
    2. Carga un modelo de predicción previamente entrenado desde un archivo `.pkl`.
    3. Preprocesa la imagen para que sea compatible con el modelo.
    4. Realiza predicciones utilizando el modelo cargado.
    5. Devuelve el número del clúster al que pertenece la imagen.

    Además, maneja errores comunes como:
    - Archivos no encontrados (modelo o imagen).
    - Errores genéricos de ejecución.

    Requisitos:
    - Tener un archivo `.pkl` con el modelo previamente entrenado.
    - La imagen debe ser preprocesada de manera que sea compatible con el modelo.
    
    Excepciones manejadas:
    - FileNotFoundError: Si el archivo del modelo o la imagen no se encuentra.
    - Exception: Cualquier otro error inesperado.
    """
    try:

        # Nombre del archivo del modelo
        archivo = "imagenes.pkl"
        modelo_path = os.path.join("recursos", "modelos", archivo)

        # Controlar que el fichero del modelo exista
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"Error: No se encontró el archivo del modelo en la ruta especificada: {modelo_path}")

        # Solicitar al usuario la imagen
        file_path = input("Por favor, introduce la ruta completa de la imagen: ")

       # Cargar y preprocesar la imagen
        imagen = Image.open(file_path).convert("L") 
        imagen = imagen.resize((64, 64))
        image_array = np.array(imagen).reshape(1, -1)

        # Cargar el modelo desde el archivo .pkl
        final_model_reloaded = joblib.load(modelo_path)
        print("Modelo cargado correctamente.")

        # Realizar la predicción usando el modelo cargado
        prediction = final_model_reloaded.predict(image_array)

        # Imprimir la predicción
        print(f"La imagen pertenece al clúster: {prediction[0]}")

    except FileNotFoundError:
        # Manejar el caso en que el archivo de la imagen no se encuentra
        print("Error: No se encontró el archivo de la imagen. Por favor, verifica el nombre del archivo.")
    except Exception as e:
        # Manejar cualquier otro error inesperado
        print(f"Se produjo un error inesperado: {e}")

if __name__ == "__main__":
    main()
