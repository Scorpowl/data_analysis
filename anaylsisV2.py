import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)
df = sns.load_dataset("titanic")

S = df[["age","fare"]].describe().T
print(S)

cat_col = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtypes in ["int64" , "float64"] ]
cat_cols = cat_col + num_but_cat

num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float"]]
# print(num_cols) 
num_cols = [col for col in num_cols if col not in cat_cols]
# print(num_cols)

def num_summary(dataframe,numerical_col,plot=False):
    quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block =True)

# num_summary(df,"age")

for col in num_cols:
    num_summary(df,col,plot=True)










