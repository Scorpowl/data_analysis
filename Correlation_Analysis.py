import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)

df = pd.read_csv("data.csv")
# print(df.iloc[:,1:-1].head())
# num_col = [col for col in df.columns if df[col].dtype in ["int64","float64"]]

# corr = df[num_col].corr()

# sns.set(rc={'figure.figsize':(12,12)})
# sns.heatmap(corr,cmap="RdBu")
# plt.show()

# Yüksek korelasyonu eleme

# cor_matrix = corr.abs()
# upper_triangel_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k = 1).astype(np.bool_))

# drop_list = [col for col in upper_triangel_matrix.columns if any(upper_triangel_matrix[col]>0.90)

def high_correlated_cols(dataframe, plot=False , corr_th = 0.90):
    num_col = [col for col in df.columns if df[col].dtype in ["int64","float64"]]
    corr = df[num_col].corr()
    cor_matrix = corr.abs() #mutlak
    upper_triangel_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k = 1).astype(np.bool_))
    drop_list = [col for col in upper_triangel_matrix.columns if any(upper_triangel_matrix[col]>corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize':(15,15)})
        sns.heatmap(corr,cmap="RdBu")   
        plt.show(block=True)
    return drop_list
#(569, 10)
droplist = high_correlated_cols(df)
high_correlated_cols(df.drop(droplist,axis=1),plot=True)