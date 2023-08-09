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

def label_encoder(dataframe,binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first = False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols,drop_first=drop_first)
    return dataframe

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def rare_analyser(dataframe, target , cat_cols):
    for col in cat_cols:
        print(col,":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                                       "RATIO": dataframe[col].value_counts() / len(dataframe),
                                       "TARGET MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtype == "O" and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

# binary_cols = [col for col in df.columns if df[col].dtype not in [int,float] and df[col].nunique() == 2]

# for col in binary_cols:
#     label_encoder(df,col)

# print(df)

# dff = load_application_train()

# binary_cols = [col for col in dff.columns if dff[col].dtype not in ["int64","float64"] and dff[col].nunique() == 2]

# for col in binary_cols:
#     label_encoder(dff,col)

# print(dff[binary_cols].head())

# a = pd.get_dummies(df, columns=["Embarked"]).head()
# binary_cols = [col for col in a.columns if a[col].dtype not in [int,float] and a[col].nunique() == 2]

# for col in binary_cols:
#     label_encoder(a,col)
# print(a)

# cat_cols, num_cols, cat_but_car = grab_col_names(df)
# ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
# print(ohe_cols)

# dff = load_application_train()
# dff["NAME_EDUCATION_TYPE"].value_counts()

# cat_cols, num_cols, cat_but_car = grab_col_names(dff)

# for i in cat_cols:
#     cat_summary(dff,i)
# rare_analyser(dff, "TARGET", cat_cols)

# new_df  =  rare_encoder(dff,0.01)
# rare_analyser(new_df, "TARGET", cat_cols)
# print(dff["OCCUPATION_TYPE"].value_counts)

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])

mms = MinMaxScaler()
df["Age_minmax_scaler"] = mms.fit_transform(df[["Age"]])

print(df.describe().T)













