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
from sklearn.model_selection import GridSearchCV


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print(("Model with rank: {0}".format(i)))
            print(("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate])))
            print(("Parameters: {0}".format(results['params'][candidate])))
            print("")


file = h5py.File('dataset/yeast_protein.h5', 'r')
X = file['X'][:]
label = file['label'][:]
file.close()

X_train, X_test, y_train, y_test = train_test_split(
    X, label, train_size=0.075, random_state=0)
del X, label
np.random.seed(777)
print("start")
#%%
# KNeighbors
clf = KNeighborsClassifier(n_jobs=-1)
kn_param_grid = {
    "n_neighbors": [5, 10, 100],
    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
    "weights": ['uniform', 'distance']
}
grid_search = GridSearchCV(clf, param_grid=kn_param_grid)
start = time()
grid_search.fit(X_train, y_train)
print((
    "GridSearchCV KNeighbors took %.2f seconds for %d candidate parameter settings."
    % (time() - start, len(grid_search.cv_results_['params']))))
report(grid_search.cv_results_)
#%%
# DecisionTree
clf = DecisionTreeClassifier()
dt_param_grid = {"criterion": ["gini", "entropy"]}
grid_search = GridSearchCV(clf, param_grid=dt_param_grid)
start = time()
grid_search.fit(X_train, y_train)
print((
    "GridSearchCV DecisionTree took %.2f seconds for %d candidate parameter settings."
    % (time() - start, len(grid_search.cv_results_['params']))))
report(grid_search.cv_results_)
#%%
# AdaBoost
clf = AdaBoostClassifier()
adb_param_grid = {
    "n_estimators": [100, 500, 1000, 1500, 2000, 3000, 5000],
    "learning_rate": [0.01, 0.1, 1, 10]
}
grid_search = GridSearchCV(clf, param_grid=adb_param_grid)
start = time()
grid_search.fit(X_train, y_train)
print((
    "GridSearchCV AdaBoost took %.2f seconds for %d candidate parameter settings."
    % (time() - start, len(grid_search.cv_results_['params']))))
report(grid_search.cv_results_)
#%%
# RandomForest
clf = RandomForestClassifier(n_jobs=-1)
rf_param_grid = {
    "n_estimators": [10, 100, 500, 1000, 1500, 2000, 3000, 5000],
    "criterion": ["gini", "entropy"]
}
grid_search = GridSearchCV(clf, param_grid=rf_param_grid)
start = time()
grid_search.fit(X_train, y_train)
print((
    "GridSearchCV RandomForest took %.2f seconds for %d candidate parameter settings."
    % (time() - start, len(grid_search.cv_results_['params']))))
report(grid_search.cv_results_)
#%%
# SVM
clf = SVC()
C_range = np.logspace(-5, 15, 10, base=2)
gamma_range = np.logspace(3, -15, 9, base=2)
svm_param_grid = {
    "C": C_range,
    "gamma": gamma_range,
    "kernel": ["rbf", "linear", "poly", "sigmoid"]
}
grid_search = GridSearchCV(clf, param_grid=svm_param_grid)
start = time()
grid_search.fit(X_train, y_train)
print((
    "GridSearchCV SVM took %.2f seconds for %d candidate parameter settings." %
    (time() - start, len(grid_search.cv_results_['params']))))
report(grid_search.cv_results_)
