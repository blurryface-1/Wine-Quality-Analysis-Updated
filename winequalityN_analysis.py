# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:25:48 2021

@author: nidhi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('winequalityN.csv')
#df.head()

# checking number of missing values
print(f"Missing values = {df.isna().sum()}")

#filling missing values with median
df['fixed acidity'] = df['fixed acidity'].fillna(df['fixed acidity'].median())
df['volatile acidity'] = df['volatile acidity'].fillna(df['volatile acidity'].median())
df['citric acid'] = df['citric acid'].fillna(df['citric acid'].median())
df['residual sugar'] = df['residual sugar'].fillna(df['residual sugar'].median())
df['chlorides'] = df['chlorides'].fillna(df['chlorides'].median())
df['pH'] = df['pH'].fillna(df['pH'].median())
df['sulphates'] = df['sulphates'].fillna(df['sulphates'].median())

df.isna().sum()

corr = df.corr()
print(f"\nCorr = \n\n{corr}")

'''
import seaborn as sns
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

'''
# Create Classification version of target variable
df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]

# for 'white' and 'red'
dummy = pd.get_dummies(df.type)
merged = pd.concat([dummy,df], axis='columns') 
final = merged.drop(['type','red'], axis='columns')
print(f"\nFinal =  \n\n{final}")

# Separate feature variables and target variable
X = final.drop(['quality','goodquality'], axis = 1)
y = final['goodquality']
X_features = df.drop(['quality','goodquality'], axis = 1)


# See proportion of good vs bad wines
print(f"\nGood quality wine: {df['goodquality'].value_counts()}") 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size=0.2, random_state=1)

# Normalize feature variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# training the model 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix = \n\n{cm}")
ac = accuracy_score(y_test, y_pred)
print(f"\nAccuracy = {ac*100}%")

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(f"\nClassification Report = \n\n{cr}")


feat_importances = pd.Series(model.feature_importances_, index = X_features.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))

# Filtering df for only good quality
df_temp = df[df['goodquality']==1]

# Filtering df for only bad quality
df_temp2 = df[df['goodquality']==0]

print(f"\nOnly good quality = \n\n{df_temp.describe()}")
print(f"\nOnly bad quality = \n\n{df_temp2.describe()}")
