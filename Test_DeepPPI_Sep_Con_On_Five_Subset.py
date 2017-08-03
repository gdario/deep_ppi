# -*- coding: utf-8 -*-
"""
Created on 2016/12/20 13:11

@author: sun
"""
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Dropout
from keras.layers.merge import concatenate

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from utils.tools import calculate_performace, categorical_probas_to_classes


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


def get_sep_model():
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


def get_con_model():
    input_1 = Input(shape=(2328, ), name='Protein')
    protein_input1 = Dense(
        512,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_feature_1')(input_1)
    protein_input1 = Dropout(0.2)(protein_input1)
    protein_input1 = Dense(
        256,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_feature_2')(protein_input1)
    protein_input1 = Dropout(0.2)(protein_input1)
    protein_input1 = Dense(
        128,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_feature_3')(protein_input1)
    protein_input1 = Dropout(0.2)(protein_input1)
    #    protein_input1 = Dense(128, activation='relu', kernel_initializer='glorot_normal', name='High_dim_feature_4')(protein_input1)
    #    protein_input1=Dropout(0.2)(protein_input1)
    output = Dense(
        128,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_feature')(protein_input1)
    outputs = Dense(2, activation='softmax', name='output')(output)
    model = Model(inputs=input_1, outputs=outputs)
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(
        loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.summary()
    return model


#%%
yeast_protein = pd.read_csv('dataset/yeast_protein_pair.csv')
yeast_feature_all = pd.read_csv('dataset/yeast_feature_all.csv', index_col=0)
# 1:1
posNum = 17257
negNum = 48594
pos_protein = yeast_protein[0:posNum]

sepscores = []
conscores = []
for j in range(5):
    #Set the random number seed
    np.random.seed(j)
    neg_protein = get_neg(posNum, negNum, yeast_protein)
    X, label = get_protein(pos_protein, neg_protein, yeast_feature_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X, label, random_state=0)

    X1_train = X_train[:, 0:1164]
    X2_train = X_train[:, 1164:2328]
    X1_test = X_test[:, 0:1164]
    X2_test = X_test[:, 1164:2328]
    del X, label, neg_protein

    model = get_sep_model()
    y_train = np_utils.to_categorical(y_train)
    hist = model.fit(
        [X1_train, X2_train], y_train, epochs=30, batch_size=64, verbose=0)

    #prediction probability
    y_score = model.predict([X1_test, X2_test])
    y_test_tmp = np_utils.to_categorical(y_test)
    fpr, tpr, _ = roc_curve(y_test_tmp[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)

    y_class = categorical_probas_to_classes(y_score)
    acc, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performace(
        len(y_class), y_class, y_test)
    print((
        'DeepPPI-sep:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,roc_auc=%f'
        % (acc, precision, npv, sensitivity, specificity, mcc, roc_auc)))
    sepscores.append(
        [acc, precision, npv, sensitivity, specificity, mcc, roc_auc])

    model = get_con_model()
    hist = model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0)

    #prediction probability
    y_score = model.predict(X_test)
    y_test_tmp = np_utils.to_categorical(y_test)
    fpr, tpr, _ = roc_curve(y_test_tmp[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)

    y_class = categorical_probas_to_classes(y_score)
    acc, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performace(
        len(y_class), y_class, y_test)
    print((
        'DeepPPI-con:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,roc_auc=%f'
        % (acc, precision, npv, sensitivity, specificity, mcc, roc_auc)))
    conscores.append(
        [acc, precision, npv, sensitivity, specificity, mcc, roc_auc])

sepscores = pd.DataFrame(sepscores)
sepscores.to_csv("sepscores.csv", index=None, header=None)
conscores = pd.DataFrame(conscores)
conscores.to_csv("conscores.csv", index=None, header=None)
