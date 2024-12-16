import pandas as pd
import joblib

def main():
    """
    Carga un modelo previamente entrenado y realiza predicciones sobre un archivo CSV proporcionado por el usuario.

    Este script realiza las siguientes acciones:
    1. Solicita al usuario el nombre de un archivo CSV que contiene los datos a analizar.
    2. Carga un modelo de predicción previamente entrenado desde un archivo `.pkl`.
    3. Valida, limpia y organiza los datos del archivo CSV para que coincidan con las características utilizadas durante el entrenamiento del modelo.
    4. Realiza predicciones utilizando el modelo cargado.
    5. Agrega las predicciones como una nueva columna al DataFrame original y las imprime.

    Además, maneja errores comunes como:
    - Archivos no encontrados (modelo o CSV).
    - Errores genéricos de ejecución.

    Requisitos:
    - Tener un archivo `.pkl` con el modelo previamente entrenado.
    - El archivo CSV debe contener las mismas columnas que las utilizadas en el entrenamiento del modelo.

    Excepciones manejadas:
    - FileNotFoundError: Si el archivo del modelo o el archivo CSV no se encuentra.
    - Exception: Cualquier otro error inesperado.

    """
    try:
        # Nombre del archivo del modelo
        archivo = "fraudes.pkl"

        # Solicitar al usuario el archivo CSV con los datos
        file_path = input("Por favor, introduce el nombre del archivo CSV con los datos: ")

        # Cargar el modelo desde el archivo .pkl
        final_model_reloaded = joblib.load("recursos/modelos/" + archivo)
        print("Modelo cargado correctamente.")

        # Lista de características utilizadas durante el entrenamiento del modelo
        feature_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                           'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                           'V26', 'V27', 'V28', 'Amount']

        # Cargar los datos desde el archivo CSV proporcionado por el usuario
        user_data = pd.read_csv("datos/" + file_path)

        # Limpiar los nombres de las columnas para evitar problemas de formato
        user_data.columns = user_data.columns.str.strip()

        # Validar y reordenar las columnas según el modelo
        user_data = user_data[feature_columns]

        # Convertir los datos a tipo numérico para asegurar compatibilidad
        user_data = user_data.astype(float)

        # Realizar la predicción usando el modelo cargado
        predictions = final_model_reloaded.predict(user_data)

        # Convertir las predicciones numéricas en texto
        prediction_text = ['Fraude' if pred == 1 else 'No fraude' for pred in predictions]

        # Crear un nuevo DataFrame con el número de transacción (índice) y la predicción
        result_df = pd.DataFrame({
            'Número de transacción': user_data.index + 1,  # Añadir 1 para empezar desde 1 en lugar de 0
            'Predicción': prediction_text
        })

        # Imprimir el resultado
        print("Predicciones del modelo:")
        print(result_df)
        
    except FileNotFoundError:
        # Manejar el caso en que el archivo del modelo o el CSV no se encuentra
        print("Error: No se encontró el archivo del modelo especificado. Por favor, verifica el nombre del archivo.")
    except Exception as e:
        # Manejar cualquier otro error inesperado
        print(f"Se produjo un error inesperado: {e}")

if __name__ == "__main__":
    main()
