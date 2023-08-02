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

def outlier_th(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up = quartile3 + iqr * 1.5
    low = quartile1 - iqr * 1.5
    return low , up

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

def grab_outliers(dataframe,col_name,index=False):
    low, up = outlier_th(df,col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] >10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        print(outlier_index)

def remove_outliers(dataframe,col_name):
    low, up = outlier_th(dataframe,col_name)
    df_without = dataframe[~((dataframe[col_name] < low) | (dataframe[col_name] > up))]
    return df_without

# df = sns.load_dataset("diamonds")
# df = df.select_dtypes(include=["float64","int64"])
# df = df.dropna()

df = load()
# print(df)

#eksik değer var mı yok mu kontrol etmek: 
# sonuc = df.isnull().values.any()
# print(sonuc)

#değişkenlerdeki eksik değer sayısı
# eksik_deger = df.isnull().sum()
# print(eksik_deger)

#dolu değer sayısını döndürür
# a = df.notnull().sum()
# print(a)

#veri setindeki eksik değer sayısı toplam
# result = df.isnull().sum().sum()
# print(result)

#en az bir tane eksik değere sahip gözlem birimleri 
# sonuc = df[df.isnull().any(axis=1)]
# print(sonuc)

# tam olan gözlem birimleri
# df[df.notnull().all(axis=1)]

# result = df.isnull().sum().sort_values(ascending=False)
# print(result)

# oran = (df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
# print(oran)

# na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
# print(na_cols)

def missing_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss =  dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio,2)] , axis=1, keys=["n_miss","ratio"])
    print(missing_df, end="\n")

    if na_name == True:
        return na_columns

#Atama yöntemi
# df["Age"].fillna(df["Age"].mean())

#Sayısal değerdeki değişkenlere atama
# df.apply(lambda x : x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

# dff = df.apply(lambda x : x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
# missing_table(dff)

#Kategorik değişkenlerde atama (modunu alabiliriz)
# df["Embarked"].fillna("missing")

# df.apply(lambda x : x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x,axis=0)

#Cinsiyete göre ortalama atama işlemi
# df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))

#loc kullanarak cinsiyete göre ortalama ataması yapma işlemi
# df.loc[(df["Age"].isnull()) & (df["Sex"] =="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
# df.loc[(df["Age"].isnull()) & (df["Sex"] =="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

# cat_cols , num_cols , cat_but_car = grab_col_names(df)
# num_cols = [col for col in num_cols if col not in "PassengerId"]

# dff = pd.get_dummies(df[cat_cols + num_cols] , drop_first=True)

# #değişkenlerin  standartlaştırılması
# scaler = MinMaxScaler()
# dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

# #KNN uygulaması
# from sklearn.impute import KNNImputer

# imputer = KNNImputer(n_neighbors=5)
# dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

# dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
# df["age_imputed_knn"] = dff[["Age"]]

# a = df.loc[df["Age"].isnull() , ["Age", "age_imputed_knn"]]
# print(a)


# print(dff.head())

def    missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"Target Mean": temp_df.groupby(col)[target].mean(),
                            "count":temp_df.groupby(col)[target].count()}), end="\n\n\n")
  

missing_table(df,True)
na_cols = missing_table(df,True)
missing_vs_target(df,"Survived", na_cols)







