#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Gabriela Fuller
#DSC 540
#I pledge on my honor that I, Gabriela Fuller, have followed the rules listed above, 
#that I have not given or received any unauthorized assistance on this assignment.


# In[2]:


import sklearn
sklearn.__version__

from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


pd.set_option('display.max_columns', 100)


df = pd.read_csv(r'C:\Users\fulle\Documents\wdbc.data', header=None)

df.columns = ['id', 'diagnosis', 'mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
df 


# In[7]:


X = df.drop(['id', 'diagnosis'], axis=1)
X.head

y = df.diagnosis
y.head()


# In[10]:


#1
df['diagnosis'].value_counts(dropna=False)



# In[11]:


df.isnull().values.any()


# In[15]:


#2

# Convert diagnosis to numerical (Malignant: 1, Benign: 0)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

#mean values
mean_features = df.iloc[:, 2:12]

# Summary
summary = mean_features.describe(percentiles=[0.25, 0.75])

# Print summary
print(summary.to_string())

# Minimum 25% 
min_q1 = summary.loc["25%", mean_features.columns].idxmin()
print(f"Feature with minimum 25th percentile: {min_q1}")

#max mean
max_mean = summary.loc["mean", mean_features.columns].idxmax()
print(f"Feature with largest mean value: {max_mean}")


# In[17]:


#3
X.corr


# In[18]:


# split the datasets into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 55, test_size= 0.25)

y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)


# In[ ]:





# In[65]:


#Question 4
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

clf = LogisticRegression(max_iter = 10000, C=0.1)

#fit the model to training data
clf.fit(X_train, y_train)

#make predictions on the test data
y_pred = clf.predict(X_test)

training_accuracy = accuracy_score(y_train, clf.predict(X_train))
testing_accuracy = accuracy_score(y_test, y_pred)

#confusion matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

recall = recall_score(y_test, y_pred, pos_label = 'M')
precision = precision_score(y_test, y_pred, pos_label = 'M')
f1 = f1_score(y_test, y_pred)
false_positive_rate = 1 - specificity_score(y_test, y_pred)

#print results
print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)
print("Confusion Matrix:\n", confusion_matrix)
print("Recall (Sensitivity):", recall)
print("Precision:", precision)
print("F1 Score:", f1)
print("False Positive Rate:", false_positive_rate)


# In[ ]:





# In[ ]:




