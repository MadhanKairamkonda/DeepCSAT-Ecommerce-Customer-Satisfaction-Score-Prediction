import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/eCommerce_Customer_support_data.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df["CSAT Score"].value_counts())

df["CSAT Score"].hist()
plt.show()