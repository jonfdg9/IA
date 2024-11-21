import pandas as pd
import numpy as np
import joblib
import sys  # Para salir del programa en caso de error crítico


def get_user_input():
    """Pide al usuario los datos necesarios para la predicción con validación de entradas."""
    print("Introduce los datos necesarios:")
    try:
        int_rate = float(input("Tasa de interés del préstamo (int_rate): "))
        installment = float(input("Cuota mensual (installment): "))
        fico = float(input("Puntaje de crédito (fico): "))
        revol_bal = float(input("Balance revolvente (revol_bal): "))
        revol_util = float(input("Porcentaje de utilización revolvente (revol_util): "))
        inq_last_6mths = int(input("Número de consultas de crédito en los últimos 6 meses (inq_last_6mths): "))
        pub_rec = int(input("Número de registros públicos de morosidad (pub_rec): "))
        purpose = input("Propósito del préstamo (purpose): ")
        credit_policy = int(input("¿Cumple con la política de crédito? (1 para Sí, 0 para No): "))
        
        # Validaciones básicas
        if int_rate < 0 or int_rate > 100:
            raise ValueError("La tasa de interés debe estar entre 0 y 100.")
        if fico < 300 or fico > 850:
            raise ValueError("El puntaje FICO debe estar entre 300 y 850.")
        if revol_util < 0 or revol_util > 100:
            raise ValueError("El porcentaje de utilización revolvente debe estar entre 0 y 100.")
        if credit_policy not in [0, 1]:
            raise ValueError("La política de crédito debe ser 0 (No) o 1 (Sí).")
        
        # Crear un diccionario con los datos
        data = {
            'int.rate': [int_rate],
            'installment': [installment],
            'fico': [fico],
            'revol.bal': [revol_bal],
            'revol.util': [revol_util],
            'inq.last.6mths': [inq_last_6mths],
            'pub.rec': [pub_rec],
            'purpose': [purpose],
            'credit.policy': [credit_policy]
        }
        return data

    except ValueError as e:
        print(f"Error en los datos ingresados: {e}")
        sys.exit(1)  # Terminar el programa en caso de error crítico


def main():
    """Función principal para cargar el modelo y predecir."""
    try:
        # Solicitar el nombre del archivo del modelo
        archivo = input("Introduce el nombre del archivo del modelo (con extensión .pkl): ")
        
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
        print(f"Predicción del modelo: {'Aprobado' if predictions[0] == 1 else 'Rechazado'}")

    except FileNotFoundError:
        print("Error: No se encontró el archivo especificado. Por favor, verifica el nombre del archivo.")
    except AttributeError as e:
        print(f"Error en el modelo: {e}")
    except Exception as e:
        print(f"Se produjo un error inesperado: {e}")


if __name__ == "__main__":
    main()
