# -*- coding: utf-8 -*-
"""
Created on 2016/12/18 8:33

@author: sun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense, Input, Dropout
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from utils.tools import calculate_performace


def get_neg(pos_num, neg_num, protein):
    index = list(range(neg_num))
    np.random.shuffle(index)
    index = np.array(index)
    neg_protein = protein.iloc[pos_num + index[0:pos_num]]
    return neg_protein


def get_protein(pos_protein_1, neg_protein_1, profeat_feature):
    pos_neg_protein = pd.concat([pos_protein_1, neg_protein_1], axis=0)
    pos_neg_protein.index = np.arange(len(pos_protein_1) * 2)
    labels = pos_neg_protein['interaction']
    protein_a = profeat_feature.loc[pos_neg_protein.proteinA, :]
    protein_b = profeat_feature.loc[pos_neg_protein.proteinB, :]
    protein_a.index = np.arange(len(pos_protein_1) * 2)
    protein_b.index = np.arange(len(pos_protein_1) * 2)
    protein = pd.concat([protein_a, protein_b], axis=1)

    #change data to numpy type
    X = np.array(protein)
    labels = np.array(labels)
    #normalization
    X = StandardScaler().fit_transform(X)
    #shuffle data
    np.random.seed(1)
    index = list(range(len(labels)))
    np.random.shuffle(index)
    X = X[index]
    labels = labels[index]
    return X, labels


def get_model():
    input_1 = Input(shape=(1164, ), name='Protein_a')
    protein_input1 = Dense(
        512,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proA_feature_1')(input_1)
    protein_input1 = Dropout(0.2)(protein_input1)
    protein_input1 = Dense(
        256,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proA_feature_2')(protein_input1)
    protein_input1 = Dropout(0.2)(protein_input1)
    protein_input1 = Dense(
        128,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proA_feature_3')(protein_input1)
    protein_input1 = Dropout(0.2)(protein_input1)
    input_2 = Input(shape=(1164, ), name='Protein_b')
    protein_input2 = Dense(
        512,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proB_feature_1')(input_2)
    protein_input2 = Dropout(0.2)(protein_input2)
    protein_input2 = Dense(
        256,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proB_feature_2')(protein_input2)
    protein_input2 = Dropout(0.2)(protein_input2)
    protein_input2 = Dense(
        128,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proB_feature_3')(protein_input2)
    protein_input2 = Dropout(0.2)(protein_input2)
    merged_vector = concatenate([protein_input1, protein_input2], axis=1)
    output = Dense(
        128,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_feature_1')(merged_vector)
    outputs = Dense(2, activation='softmax', name='output')(output)
    model = Model(inputs=[input_1, input_2], outputs=outputs)
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(
        loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.summary()
    return model


yeast_protein = pd.read_csv('dataset/yeast_protein_pair.csv')
yeast_profeat_all = pd.read_csv('dataset/yeast_feature_all.csv', index_col=0)
# 1:1
posNum = 17257
negNum = 48594
pos_protein = yeast_protein[0:posNum]

cvscores = []
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
lw = 2
i = 0

#Set the random number seed
np.random.seed(0)
neg_protein = get_neg(posNum, negNum, yeast_protein)
X, label = get_protein(pos_protein, neg_protein, yeast_profeat_all)

X_train, X_test, y_train, y_test = train_test_split(X, label, random_state=0)

X1_train = X_train[:, 0:1164]
X2_train = X_train[:, 1164:2328]
X1_test = X_test[:, 0:1164]
X2_test = X_test[:, 1164:2328]
del X, label, neg_protein

model = get_model()
y_train = np_utils.to_categorical(y_train)
model.fit([X1_train, X2_train], y_train, epochs=30, batch_size=64, verbose=0)

#prediction probability
y_score = model.predict([X1_test, X2_test])
y_test = np_utils.to_categorical(y_test)
fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 0])
roc_auc = auc(fpr, tpr)

plt.plot(
    fpr,
    tpr,
    lw=lw,
    color=plt.cm.Set1(i / 10.),
    label='ROC DNN (area = %0.2f%%)' % (roc_auc * 100))
i += 1

y_score = np_utils.categorical_probas_to_classes(y_score)
y_test = np_utils.categorical_probas_to_classes(y_test)
acc, precision, npv, sensitivity, specificity, mcc = calculate_performace(
    len(y_score), y_score, y_test)
print((
    'DeepPPI:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,roc_auc=%f'
    % (acc, precision, npv, sensitivity, specificity, mcc, roc_auc)))
cvscores.append([acc, precision, npv, sensitivity, specificity, mcc, roc_auc])

names = [
    "Nearest Neighbors", "SVM", "Decision Tree", "Random Forest", "AdaBoost",
    "Naive Bayes", "QDA"
]
classifiers = [
    KNeighborsClassifier(n_neighbors=10, weights="distance", algorithm="auto"),
    SVC(probability=True, C=3.1748021039363996, gamma=0.00069053396600248786),
    DecisionTreeClassifier(criterion="entropy"),
    RandomForestClassifier(n_estimators=3000),
    AdaBoostClassifier(n_estimators=3000),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

y_train = np_utils.categorical_probas_to_classes(y_train)
# scikit-learning
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)

    y_temp = np_utils.to_categorical(y_test)
    fpr, tpr, _ = roc_curve(y_temp[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        lw=lw,
        color=plt.cm.Set1(i / 7.),
        label='ROC %s (area = %0.2f%%)' % (name, (roc_auc * 100)))
    i += 1

    y_score = np_utils.categorical_probas_to_classes(y_score)

    acc, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performace(
        len(y_score), y_score, y_test)
    print((
        '%s:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,roc_auc=%f'
        % (name, acc, precision, npv, sensitivity, specificity, mcc, roc_auc)))
    cvscores.append(
        [acc, precision, npv, sensitivity, specificity, mcc, roc_auc])

plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
