import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler,RobustScaler

pd.set_option("display.max_columns",None)

def load_application_train():
    data = pd.read_csv("application_train.csv")
    return data

def load():
    data = pd.read_csv("titanic.csv")
    return data

df = load()
dff = load_application_train()
# print(dff)

# q1 = df["Age"].quantile(0.25) #20.125
# q3 = df["Age"].quantile(0.75) #38.0
# iqr = q3 - q1 #17.875
# upper = q3 + 1.5 * iqr #64.8125
# lower = q1 - 1.5 * iqr #-6.6875



def outlier_th(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up = quartile3 + iqr * 1.5
    low = quartile1 - iqr * 1.5
    return up , low

def check_outlier(dataframe,col_name):
    up , low = outlier_th(dataframe,col_name)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
        return True
    else:
        return False

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    num_cols = [col for col in num_cols if col not in "PassengerId"]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len (num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

for col in num_cols:
    print(col, check_outlier(dff,col))





























