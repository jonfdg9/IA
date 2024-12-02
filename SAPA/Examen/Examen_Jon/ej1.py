import pandas as pd
import numpy as np
import joblib
import sys  # Para salir del programa en caso de error crítico


def get_user_input():
    """Pide al usuario los datos necesarios para la predicción con validación de entradas."""
    print("Introduce los datos necesarios:")
    try:
        anc_petalo = float(input("Escribe la anchura del pétalo: "))
        lar_petalo = float(input("Escribe la largura del pétalo: "))
        anc_sepalo = float(input("Escribe la anchura del sépalo: "))
        lar_sepalo = float(input("Escribe la largura del sépalo: "))

        # Validaciones básicas
        if (anc_petalo <= 0.0 ):
            raise ValueError("El valor introducido tiene que se mayor a 0.")
        if (lar_petalo <= 0.0 ):
            raise ValueError("El valor introducido tiene que se mayor a 0.")
        if (anc_sepalo <= 0.0 ):
            raise ValueError("El valor introducido tiene que se mayor a 0.")
        if (lar_sepalo <= 0.0 ):
            raise ValueError("El valor introducido tiene que se mayor a 0.")
        
        # Crear un diccionario con los datos
        data = {
            'sepal length (cm)': [lar_sepalo],
            'sepal width (cm)': [anc_sepalo],
            'petal length (cm)': [lar_petalo],
            'petal width (cm)': [anc_petalo],             
        }
        return data

    except ValueError as e:
        print(f"Error en los datos ingresados: {e}")
        sys.exit(1)  # Terminar el programa en caso de error crítico


def main():
    """Función principal para cargar el modelo y predecir."""
    try:
        # Nombre del archivo
        archivo = "flores.pkl"
        
        # Cargar el modelo
        final_model_reloaded = joblib.load("SAPA/Examen/Examen_Jon/"+archivo)
        
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

        flores = ['Setosa', 'Versicolor', 'Virginica']
        print(f"Predicción del modelo: {flores[predictions[0]]}")

    except FileNotFoundError:
        print("Error: No se encontró el archivo especificado. Por favor, verifica el nombre del archivo.")
    except AttributeError as e:
        print(f"Error en el modelo: {e}")
    except Exception as e:
        print(f"Se produjo un error inesperado: {e}")


if __name__ == "__main__":
    main()
