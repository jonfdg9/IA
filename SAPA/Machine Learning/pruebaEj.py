import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer , make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics import accuracy_score
import joblib
import warnings
 
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
    bins = [-1, 16, 32, 48, 64, 100]
    labels = [1, 2, 3, 4, 5]

    categorized_age = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        age = X[i, 0] 
        if age <= bins[1]:
            categorized_age[i] = labels[0]
        elif age <= bins[2]:
            categorized_age[i] = labels[1]  
        elif age <= bins[3]:
            categorized_age[i] = labels[2]
        elif age <= bins[4]:
            categorized_age[i] = labels[3]
        else:
            categorized_age[i] = labels[4]    
    return categorized_age.reshape(-1, 1)

#    [...] Todas las funciones definidas por nosotros
df_titanic = sns.load_dataset('titanic')
final_model_reloaded = joblib.load("/home/iabd/Escritorio/IA/SAPA/modeloTitanic.pkl")
new_data = df_titanic[:5]  
predictions = final_model_reloaded.predict(new_data)
print(predictions)