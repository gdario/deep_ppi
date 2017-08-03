# -*- coding: utf-8 -*-
"""
Created on 2016/12/18 8:33

@author: sun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import utils.tools as utils

file = h5py.File('dataset/yeast_protein.h5', 'r')
X = file['X'][:]
label = file['label'][:]
file.close()

X_train, X_test, y_train, y_test = train_test_split(X, label, random_state=0)
del X, label
np.random.seed(777)
print("start")

cvscores = []

names = [
    "Naive Bayes", "QDA", "Nearest Neighbors", "Decision Tree",
    "Random Forest", "AdaBoost", "SVM"
]
classifiers = [
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(n_neighbors=10, weights="distance", algorithm="auto"),
    DecisionTreeClassifier(criterion="entropy"),
    RandomForestClassifier(n_estimators=3000),
    AdaBoostClassifier(n_estimators=3000),
    SVC(probability=True, C=3.1748021039363996, gamma=0.00069053396600248786),
]

# scikit-learning
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)

    y_temp = utils.to_categorical(y_test)
    fpr, tpr, _ = roc_curve(y_temp[:, 0], y_score[:, 0])

    tpr_fpr = pd.DataFrame([fpr, tpr]).T
    tpr_fpr.to_csv(name + '_tpr_fpr.csv', header=None, index=None)

    roc_auc = auc(fpr, tpr)
    y_class = utils.categorical_probas_to_classes(y_score)

    acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(
        len(y_class), y_class, y_test)
    print((
        '%s:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,roc_auc=%f'
        % (name, acc, precision, npv, sensitivity, specificity, mcc, roc_auc)))
    cvscores.append(
        [acc, precision, npv, sensitivity, specificity, mcc, roc_auc])

cvindex = pd.DataFrame(cvscores)
cvindex.to_csv("index.csv", index=None, header=None)
