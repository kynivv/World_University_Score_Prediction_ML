import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import explained_variance_score as evs
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor

# Data Import
df = pd.read_csv('World University Rankings 2023.csv')


# Data Preparation
print(df.isnull().sum())

print(df.columns)

print(df.dtypes)

df.dropna(inplace= True)

print(df.isnull().sum())

for col in df.columns:
    le = LabelEncoder()
    
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print(df.dtypes)


# Train Test Split
X = df.drop('OverAll Score', axis= 1)
Y = df['OverAll Score']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size= 0.25,
                                                    random_state= 24,
                                                    shuffle= True
                                                    )


# Model Training
m = XGBRegressor()

m.fit(X_train, Y_train)

pred_train = m.predict(X_train)
print(f'Train Accuracy is : {evs(Y_train, pred_train)}')

pred_test = m.predict(X_test)
print(f'Test Accuracy is : {evs(Y_test, pred_test)}')