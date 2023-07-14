import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)
df = sns.load_dataset("titanic")
df1 = sns.load_dataset("car_crashes")

def check_df(dataframe,head=5):

    print("SHAPE")
    print(dataframe.shape)
    print("TYPE")
    print(dataframe.dtypes)
    print("HEAD")
    print(dataframe.head(head))
    print("TAİL")
    print(dataframe.tail(head))


cat_col = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtypes in ["int64" , "float64"] ]
cat_cols = cat_col + num_but_cat

def cat_summary(dataframe, colname, plot = False):
    print("****************************************************************")

    print(pd.DataFrame({colname : dataframe[colname].value_counts(),
                        "Ratio" : 100 * dataframe[colname].value_counts() / len(dataframe) }))
    print("****************************************************************")

    if plot:
        sns.countplot(x = dataframe[colname], data = dataframe)
        plt.title(colname.upper())
        plt.show(block = True)
        
for col in cat_cols:
    if df[col].dtypes == "bool":
        print("asjhjkdgkjhgjkhfjhfdhjk")
        df[col] = df[col].astype(int)
        cat_summary(df,col, plot= True)
    else:
        cat_summary(df,col, plot= True)



