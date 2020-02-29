#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
import pandas as pd
import seaborn as sn
import sklearn
import sklearn.tree as tree
import sklearn.ensemble as ensemble
from sklearn.datasets import make_moons
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


# Ref:
# https://scikit-learn.org/stable/supervised_learning.html
# 
# use more Ensemble learning algos

# # Get all SAL files and create 1 df

# In[ ]:

print("Start")

import os
from pprint import pprint

all_files = [f for f in os.walk('/lustre/scratch/tv349/AMR/BinaryPTrainingData/')]

SAL_4k_paths = []
for path, folders, files in all_files:
    if "SAL" in path and "k4" in path:
        SAL_4k_paths.append(path)

for path in SAL_4k_paths:
    sal_name = path.rsplit("/", 2)[1]
    
    if sal_name == "SALKAN":
        f0 = f"{path}/TRAININGDATA.{sal_name}.xlsx K4-Part0.p"
        f1 = f"{path}/TRAININGDATA.{sal_name}.xlsx K4-Part1.p"
        f2 = f"{path}/TRAININGDATA.{sal_name}.xlsx K4-Part2.p"

        df0 = pd.read_pickle(f0)
        df1 = pd.read_pickle(f1)
        df2 = pd.read_pickle(f2)
    
# df = df0 + df1 + df2
df = df0.append(df1, ignore_index=True)
df = df.append(df2, ignore_index=True)

print("Got all the file")

# In[3]:


# for 4 mers
X = df.drop(columns=["Antibiotic", "MIC"], axis=1) # pick every but (Antibiotic, MIC)
y = df.iloc[:,1] # pick 2nd column (MIC values)


# In[8]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test_up = pd.DataFrame([str(float(x) + 1) for x in y_test]).iloc[:, 0]
y_test_down = pd.DataFrame([str(float(x) - 1) for x in y_test]).iloc[:, 0]


# # Confusion matrix for 2 dilutions

# In[39]:


def create_confusion_matrix(y_test, y_pred):
    # Get the difference between pred and test data
    y_test_numpy = y_test.values
    diff = y_pred.astype(float) - y_test_numpy.astype(float)

    # only keep +- 1 dilutions
    diff[diff > 1]=0
    diff[diff < -1]=0

    # force pred values to match +- 1 actual values 
    y_pred2 = y_pred.astype(float) - diff

    # Confusion matrix
    labels = [float(val) for val in sorted(set(y_test_numpy) | set(y_pred))]
    return sklearn.metrics.confusion_matrix(y_test_numpy.astype(float), y_pred2, labels=labels), labels


print("Start Naive Bayes")
# # Naive Bayes

# In[10]:


clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Prediction
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score}")

# 2 dilutions Prediction
score += accuracy_score(y_test_up, y_pred)
score += accuracy_score(y_test_down, y_pred)
print(f"Accuracy (2 dilutions): {score}")

# Confusion matrix
print(create_confusion_matrix(y_test, y_pred))

print("Start KNN")
# # KNN

# In[40]:


clf = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Prediction
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score}")

# 2 dilutions Prediction
score += accuracy_score(y_test_up, y_pred)
score += accuracy_score(y_test_down, y_pred)
print(f"Accuracy (2 dilutions): {score}")

# Confusion matrix
print(create_confusion_matrix(y_test, y_pred))


# # K means.
# Noted that this is unsupervised and we need to provide number of classes which can be found from `len(set(MIC column))`
print("Start SVM")
# # SVM

# In[12]:


clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Prediction
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score}")

# 2 dilutions Prediction
score += accuracy_score(y_test_up, y_pred)
score += accuracy_score(y_test_down, y_pred)
print(f"Accuracy (2 dilutions): {score}")

# Confusion matrix
print(create_confusion_matrix(y_test, y_pred))

print("Start Decision tree")
# # Decision tree

# In[14]:


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Prediction
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score}")

# 2 dilutions Prediction
score += accuracy_score(y_test_up, y_pred)
score += accuracy_score(y_test_down, y_pred)
print(f"Accuracy (2 dilutions): {score}")

# Confusion matrix
print(create_confusion_matrix(y_test, y_pred))

print("Start Random Forest")
# # Random Forest

# In[17]:


clf = ensemble.RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Prediction
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score}")

# 2 dilutions Prediction
score += accuracy_score(y_test_up, y_pred)
score += accuracy_score(y_test_down, y_pred)
print(f"Accuracy (2 dilutions): {score}")

# Confusion matrix
print(create_confusion_matrix(y_test, y_pred))

print("Start AdaBoost")
# # AdaBoost

# In[19]:


clf = ensemble.AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Prediction
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score}")

# 2 dilutions Prediction
score += accuracy_score(y_test_up, y_pred)
score += accuracy_score(y_test_down, y_pred)
print(f"Accuracy (2 dilutions): {score}")

# Confusion matrix
print(create_confusion_matrix(y_test, y_pred))

print("Start Gradient Boosting")
# # Gradient Boosting

# In[21]:


clf = ensemble.GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Prediction
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score}")

# 2 dilutions Prediction
score += accuracy_score(y_test_up, y_pred)
score += accuracy_score(y_test_down, y_pred)
print(f"Accuracy (2 dilutions): {score}")

# Confusion matrix
print(create_confusion_matrix(y_test, y_pred))

