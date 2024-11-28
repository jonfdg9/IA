import pandas as pd
import numpy as np
import joblib
import sys  # Para salir del programa en caso de error crítico


def get_user_input():
    """Pide al usuario los datos necesarios para la predicción de CO2 del coche."""
    print("Introduce los datos necesarios:")
    try:
        volume = float(input("Volume: "))
        weight = float(input("Weight: "))
          
        # Validaciones básicas
        if volume < 0:
            raise ValueError("El volumen tiene que ser positivo.")
        if weight < 0:
            raise ValueError("El peso tiene que ser positivo.")    
        # Crear un diccionario con los datos
        data = {
            'Volume': [volume],
            'Weight': [weight],
        }
        return data

    except ValueError as e:
        print(f"Error en los datos ingresados: {e}")
        sys.exit(1)  # Terminar el programa en caso de error crítico


def main():
    """Función principal para cargar el modelo y predecir."""
    try:
        # Nombre del archivo
        archivo = "co2.pkl"
        
        # Cargar el modelo
        final_model_reloaded = joblib.load("/home/iabd/Escritorio/IA/SAPA/Machine Learning/"+archivo)
        
        # Verificar si el modelo cargado tiene un método predict
        if not hasattr(final_model_reloaded, "predict"):
            raise AttributeError("El archivo cargado no contiene un modelo válido con el método predict.")
        
        print("Modelo cargado correctamente.")
        
        # Obtener los datos del usuario
        user_data = get_user_input()

        # Convertir los datos en un DataFrame
        user_df = pd.DataFrame(user_data)

        # Hacer la predicción
        predictions = final_model_reloaded.predict(user_df)
        
        # Mostrar la predicción
        print(f"Predicción del modelo: { predictions[0]}")

    except FileNotFoundError:
        print("Error: No se encontró el archivo especificado. Por favor, verifica el nombre del archivo.")
    except AttributeError as e:
        print(f"Error en el modelo: {e}")
    except Exception as e:
        print(f"Se produjo un error inesperado: {e}")


if __name__ == "__main__":
    main()
