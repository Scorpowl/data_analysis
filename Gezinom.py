import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",None)
df = pd.read_excel("miuul_gezinomi.xlsx")
# print(df)

#veri setinin bilgisi
# print(df.shape)

#unique şehir sayısı
# print(df["SaleCityName"].nunique())

#Şehirler nelerdir?
# print(df["SaleCityName"].unique())

#Hangi şehirden kaç tane var?
# print(df["SaleCityName"].value_counts())

#Kaç unique concept vardır?
# print(df["ConceptName"].nunique())

#Hangi concept ten kaçar tane satış gerçekleşmiştir?
# print(df["ConceptName"].value_counts())

#Şehirlere göre satışlardan toplam ne kadar kazanıldı?
# print(df.groupby("SaleCityName").agg({"Price":"sum"}))

#Concept türlerine göre göre ne kadar kazanılmış?
# print(df.groupby("ConceptName").agg({"Price":"sum"}))

#Şehirlere göre PRICE ortalamaları nedir?
# print(df.groupby("SaleCityName").agg({"Price":"mean"}))

#Conceptlere  göre PRICE ortalamaları nedir?
# print(df.groupby("ConceptName").agg({"Price":"mean"}))

#Şehir-Concept kırılımında PRICE ortalamaları nedir?
# print(df.groupby(["SaleCityName","ConceptName"]).agg({"Price":"mean"}))

## GÖREV 2: satis_checkin_day_diff değişkenini EB_Score adında yeni bir kategorik değişkene çeviriniz.
# bins = [-1,7,30,90,df["SaleCheckInDayDiff"].max()]
# labels = ["Last Minuters", "Potential Planners", "Planners", "Early Bookers"]
# df["EB_Score"] = pd.cut(df["SaleCheckInDayDiff"],bins,labels=labels)
# df.head(50).to_excel("eb_score.xlsx",index=False)

## GÖREV 3: Şehir,Concept, [EB_Score,Sezon,CInday] kırılımında ücret ortalamalarına ve frekanslarına bakınız
# Şehir-Concept-EB Score kırılımında ücret ortalamaları
# print(df.groupby(["SaleCityName","ConceptName","EB_Score"]).agg({"Price":["mean","count"]}))
# Şehir-Concept-Sezon kırılımında ücret ortalamaları
# print(df.groupby(["SaleCityName","ConceptName","Sezon"]).agg({"Price":["mean","count"]}))
# Şehir-Concept-CInday kırılımında ücret ortalamaları
# print(df.groupby(["SaleCityName","ConceptName","CInday"]).agg({"Price":["mean","count"]}))

# GÖREV 4: City-Concept-Season kırılımın çıktısını PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

agg_df = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False)

# GÖREV 5: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()
agg_df.reset_index(inplace=True)

agg_df.head()
#############################################
# GÖREV 6: Yeni level based satışları tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# sales_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
agg_df['sales_level_based'] = agg_df[["SaleCityName", "ConceptName", "Seasons"]].agg(lambda x: '_'.join(x).upper(), axis=1)


#############################################
# GÖREV 7: Personaları segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz
# segmentleri betimleyiniz
agg_df["SEGMENT"] = pd.qcut(agg_df["Price"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"Price": ["mean", "max", "sum"]})

#############################################
# GÖREV 8: Oluşan son df'i price değişkenine göre sıralayınız.
# "ANTALYA_HERŞEY DAHIL_HIGH" hangi segmenttedir ve ne kadar ücret beklenmektedir?
#############################################
agg_df.sort_values(by="Price")


new_user = "ANTALYA_HERŞEY DAHIL_HIGH"
agg_df[agg_df["sales_level_based"] == new_user]











