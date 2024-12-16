import joblib
import numpy as np

modelo = joblib.load("Mejor_RFC.pkl")

anc_petalo = float(input("Escribe la anchura del pétalo: "))
lar_petalo = float(input("Escribe la largura del pétalo: "))
anc_sepalo = float(input("Escribe la anchura del sépalo: "))
lar_sepalo = float(input("Escribe la largura del sépalo: "))

a_estimar = np.array([lar_sepalo, anc_sepalo, lar_petalo, anc_petalo]).reshape(1, -1)

predicción = modelo.predict(a_estimar)

target_names = ['setosa', 'versicolor', 'virginica']

print(f"El tipo de flor es {target_names[predicción[0]]}.")