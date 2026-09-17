import numpy as np
import pandas as pd
import seaborn as scs
import matplotlib.pyplot as mpt

df = pd.read_csv("insurance.csv")

# EDA

# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.head())
# print(df.isnull().sum())
# print(df.columns)

numericl_column = ['age', 'bmi', 'children', 'charges']

# for col in numericl_column:
#     mpt.figure(figsize=(8,6))
#     scs.histplot(df[col],kde = True,bins=20)
# print(numericl_column)
# mpt.show()

scs.countplot(x = df['children'])
mpt.show()

scs.countplot(x= df["sex"])
mpt.show()

scs.countplot(x= df['smoker'])
mpt.show()

for col in numericl_column:
    mpt.figure(figsize=(6,4))
    scs.boxplot(x= df[col])

mpt.show()

mpt.figure(figsize=(8,6))
scs.heatmap(df.corr(numeric_only=True), annot=True)
mpt.show()


# DATA CLEANING AND PREPROCESSING