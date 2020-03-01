#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
import pandas as pd
# import seaborn as sn
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
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# Ref:
# https://scikit-learn.org/stable/supervised_learning.html
# 
# use more Ensemble learning algos


import os
from pprint import pprint

all_files = [f for f in os.walk('/lustre/scratch/tv349/AMR/BinaryPTrainingData/')]

SAL_4k_paths = []
for path, folders, files in all_files:
    if "SAL" in path and "k4" in path:
        SAL_4k_paths.append(path)

# ("SALCIP","SALFIS","SALNAL")
# ("SALAUG","SALCOT","SALSTR")
# ("SALAXO","SALGEN","SALTET")
# ("SALCHL","SALAMP","SALFOX")
# ("SALAZI","SALTIO","SALKAN")
for path in SAL_4k_paths:
    sal_name = path.rsplit("/", 2)[1]
    if sal_name in ("SALAXO","SALGEN","SALTET","SALCHL","SALAMP","SALFOX"):
        print(f"Start {sal_name}")
        f0 = f"{path}/TRAININGDATA.{sal_name}.xlsx K4-Part0.p"
        f1 = f"{path}/TRAININGDATA.{sal_name}.xlsx K4-Part1.p"
        f2 = f"{path}/TRAININGDATA.{sal_name}.xlsx K4-Part2.p"

        df0 = pd.read_pickle(f0)
        df1 = pd.read_pickle(f1)
        df2 = pd.read_pickle(f2)

        # df = df0 + df1 + df2
        df = df0.append(df1, ignore_index=True)
        df = df.append(df2, ignore_index=True)
        print(f"Got all the file")

        # There are unsual case where MIC column has values like 6.65555555 or 5.088888. SO i change to float then round it then change it to str.  
        df['MIC'] = df['MIC'].astype(float).round(0).astype(str)

        # for 4 mers
        X = df.drop(columns=["Antibiotic", "MIC"], axis=1) # pick every but (Antibiotic, MIC)
        y = df.iloc[:,1] # pick 2nd column (MIC values)

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_test_up = pd.DataFrame([str(float(x) + 1) for x in y_test]).iloc[:, 0]
        y_test_down = pd.DataFrame([str(float(x) - 1) for x in y_test]).iloc[:, 0]

        all_data_for_saving = [sal_name]
        all_data_for_saving.append([])

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
            rv = sklearn.metrics.confusion_matrix(y_test_numpy.astype(float), y_pred2, labels=labels), labels
            all_data_for_saving[-1].extend(rv)
            return rv


        def print_acc_cfm(y_test, y_pred, name):
            all_data_for_saving[-1].append(name)

            # Prediction
            score = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {score}")
            all_data_for_saving[-1].append(f"Accuracy: {score}")

            # 2 dilutions Prediction
            score += accuracy_score(y_test_up, y_pred)
            score += accuracy_score(y_test_down, y_pred)
            print(f"Accuracy (2 dilutions): {score}")
            all_data_for_saving[-1].append(f"Accuracy (2 dilutions): {score}")

            # Confusion matrix
            print(create_confusion_matrix(y_test, y_pred))

        print("Start Naive Bayes")
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print_acc_cfm(y_test, y_pred, "Naive Bayes")
        
        print("Start KNN")
        clf = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print_acc_cfm(y_test, y_pred, "KNN")

        print("Start SVM")
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print_acc_cfm(y_test, y_pred, "SVM")

        print("Start Decision tree")
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print_acc_cfm(y_test, y_pred, "Decision Tree")

        print("Start Random Forest")
        clf = ensemble.RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print_acc_cfm(y_test, y_pred, "Random Forest")

        print("Start AdaBoost")
        clf = ensemble.AdaBoostClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print_acc_cfm(y_test, y_pred, "AdaBoost")

        print("Start Gradient Boosting")
        clf = ensemble.GradientBoostingClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print_acc_cfm(y_test, y_pred, "GradientBoost")

        try:
            current_time = datetime.now().strftime("%H-%M-%S_%m-%d-%Y")
            with open(f"results/{sal_name}_{current_time}.log", "w") as f:
                f.writelines(f"{val}\n" for val in all_data_for_saving)
        except:
            print(f"Error with saving data for {sal_name}", e)