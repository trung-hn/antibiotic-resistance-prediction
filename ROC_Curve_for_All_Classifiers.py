#!/usr/bin/env python
# coding: utf-8



import pickle
import sys
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
from sklearn.model_selection import KFold
# START For ROC Curve Tools
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
from sklearn.model_selection import train_test_split
# END For ROC Curve Tools
# from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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


try:
    os.mkdir("./Results")
except: pass

# FUNCTION DEF
def get_2_dil_acc_cfm(y_test, y_pred, labels):
    # Get the difference between pred and test data
    diff = y_pred.astype(float) - y_test.astype(float)

    # only keep +- 1 dilutions
    diff[diff > 1]=0
    diff[diff < -1]=0

    # force pred values to match +- 1 actual values 
    y_pred_2_dil = y_pred.astype(float) - diff

    # Calculate 2-dilution acc
    score = sum(y_test.astype(float) == y_pred_2_dil) / len(y_test)

    # Confusion matrix
    return score, sklearn.metrics.confusion_matrix(y_test.astype(float), y_pred_2_dil, labels=labels)


def plot_roc_curve_multiclass(clf, X, y, sal_name, name, labels):
    # Binarize output
    y = label_binarize(y.astype("float"), classes=list(labels))
    n_classes = y.shape[1]

    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train Classifier
    classifier = OneVsRestClassifier(clf)
    if name == "Adaboost":
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    else:
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC Curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC Curve for a specific class
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], label='Micro-Average ROC Curve (area = %.2f)' % (roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label='Macro-Average ROC Curve (area = %.2f)' % (roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
    #colors = cycle(['darkorange', 'cornflowerblue', 'darkkhaki', 'crimson'])
    for k in range(n_classes):
        plt.plot(fpr[k], tpr[k], lw=lw, label='ROC Curve of class %.1f (Area = %.2f)' % (k, roc_auc[k]))
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for %s with %s' % (sal_name, name))
    plt.legend(loc="lower right")
    plt.savefig('./Results/ROC_%s_%s.png' % (sal_name, name))


def k_fold_calculation(clf, X, y, name, labels):
    all_data_for_saving.append(name)
    acc_avg = 0
    kf = KFold(n_splits=10)
    for index, (train, test) in enumerate(kf.split(X,y=y)):
        X_train = X.values[train,:]
        y_train = y.values[train]

        X_test = X.values[test,:]
        y_test = y.values[test]

        clf.fit(X_train, y_train)
        # save model
        #filename = "/scratch/ars443/Classifiers/" + name + str(index) + ".clf"
        #try:
        #    pickle.dump(clf, open(filename, 'wb'))
        #except:
        #    print('Cant save classifier to file: ', filename)
        
        y_pred = clf.predict(X_test)

        score, cfm = get_2_dil_acc_cfm(y_test, y_pred, labels)
        acc_avg += score/10
        #print(f"Validation index {index}: Accuracy (2 dilutions): {score}")
        #print(cfm)
        all_data_for_saving.append(f"Validation index {index}: Accuracy (2 dilutions): {score}")
        all_data_for_saving.append(cfm)
        
        if index == 0:
            avg_cfm = cfm
        else:
            avg_cfm = avg_cfm + cfm
    #print("Average stats:")
    #print(f"Accuracy (2 dilutions): {acc_avg}")
    #print(avg_cfm//10)    
    all_data_for_saving.append(f"Accuracy (2 dilutions): {acc_avg}")
    all_data_for_saving.append(avg_cfm//10)
    all_data_for_saving.append(labels)
    
all_data_for_saving = []
    
antibiotics = ["SALKAN"]
if len(sys.argv) > 1:
    antibiotics = sys.argv[1:]

print(antibiotics)
for path in SAL_4k_paths:
    sal_name = path.rsplit("/", 2)[1]
    if sal_name in antibiotics:
        all_data_for_saving = [sal_name]
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
       
        # PCA of input data
        #pca = PCA(random_state=42)
        #pca.fit(X)
        #var = pca.explained_variance_ratio_
        #plt.plot((np.cumsum(var)))
        #plt.title(f"PCA for {sal_name}")
        #plt.savefig(f"./Results/{sal_name}_pca.png")
        
        # get all unique labels
        labels = set(y.astype("float"))
        labels = sorted(labels)        
        
        #print("Start Naive Bayes")
        #clf = GaussianNB()
        #k_fold_calculation(clf, X, y, "Naive Bayes", labels)
        
        #print("Start KNN")
        #clf = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=2)
        #k_fold_calculation(clf, X, y, "KNN", labels)

        # print("Start SVM")
        # clf = svm.SVC()
        # k_fold_calculation(clf, X, y, "SVM", labels)
        
        # print("Start Decision tree")
        # clf = tree.DecisionTreeClassifier()
        # k_fold_calculation(clf, X, y, "Decision tree", labels)
        
        print("Start Random Forest")
        clf = ensemble.RandomForestClassifier(n_estimators=200, max_features="auto",random_state=0)
        #k_fold_calculation(clf, X, y, "Random Forest", labels)
        plot_roc_curve_multiclass(clf, X, y, sal_name, "Random Forest", labels)

        #print("Start AdaBoost")
        #clf = ensemble.AdaBoostClassifier(n_estimators=100)
        #k_fold_calculation(clf, X, y, "Adaboost", labels)
        #plot_roc_curve_multiclass(clf, X, y, sal_name, "Adaboost", labels)

        # print("Start Gradient Boosting")
        # clf = ensemble.GradientBoostingClassifier(n_estimators=100)
        # k_fold_calculation(clf, X, y, "GradientBoost", labels)

        try:
            current_time = datetime.now().strftime("%H-%M-%S_%m-%d-%Y")
            with open(f"./Results/{sal_name}_{current_time}.log", "w") as f:
                f.write("\n".join(str(val) for val in all_data_for_saving))
        except:
            print(f"Error with saving data for {sal_name}", e)
