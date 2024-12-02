import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

final_model_reloaded = joblib.load("modelo_iris.pkl")

petal_l = int(input("Petal lenght: "))
petal_w = int(input("Petal width: "))
sepal_l = int(input("Sepal lenght: "))
sepal_w = int(input("Sepal width: "))

data = {"sepal length (cm)": [sepal_l],
        "sepal width (cm)": [sepal_w],
        "petal length (cm)": [petal_l],
        "petal width (cm)": [petal_w],
    }

df = pd.DataFrame(data=data, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

predictions = final_model_reloaded.predict(df)
print(predictions[0])
