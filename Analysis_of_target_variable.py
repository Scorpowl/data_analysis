import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)
df = sns.load_dataset("titanic")

cat_col = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtypes in ["int64" , "float64"] ]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]
cat_cols = cat_col + num_but_cat

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def grab_col_names(dataframe, cat_th=10, car_th=30):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin ismini verir.
    Args:
        dataframe (dataframe): alınmak istenen dataframedir.
        cat_th (int, optional): numerik fakat kategorik. Defaults to 10.
        car_th (int, optional): kategorik fakat kardinal. Defaults to 30.
    Returns
        cat_cols : list
        num_cols : list
        cat_but_car : list
    """
    cat_col = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtypes in ["int64" , "float64"] ]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]
    cat_cols = cat_col + num_but_cat
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len (num_but_cat)}')

    return cat_col , num_cols , cat_but_car

# cat_col , num_cols , cat_but_car = grab_col_names(df)

def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({"Target Mean":dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df,"survived",col)




