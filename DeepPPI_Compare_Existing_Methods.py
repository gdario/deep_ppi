# -*- coding: utf-8 -*-
"""
Created on Sun Jan 01 18:15:26 2017

@author: sun
"""
import numpy as np
import pandas as pd
from keras.layers import Dense, merge, Input, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from utils.tools import  calculate_performace, draw_roc,draw_pr, \
    categorical_probas_to_classes


def get_dataset(feature_file, protein_file, posnum, negnum):
    feature = pd.read_csv(feature_file, index_col=0)
    protein = pd.read_csv(protein_file)

    protein = feature.loc[protein.id, :]

    P_protein_a = protein.iloc[0:posnum, :]
    P_protein_b = protein.iloc[posnum:posnum * 2, :]
    P_protein_a.index = list(range(posnum))
    P_protein_b.index = list(range(posnum))
    pos_protein = pd.concat([P_protein_a, P_protein_b], axis=1)

    N_protein_a = protein.iloc[posnum * 2:posnum * 2 + negnum, :]
    N_protein_b = protein.iloc[posnum * 2 + negnum:posnum * 2 + negnum * 2, :]
    N_protein_a.index = list(range(negnum))
    N_protein_b.index = list(range(negnum))
    neg_protein = pd.concat([N_protein_a, N_protein_b], axis=1)

    pos_label = pd.DataFrame(np.ones(posnum))
    neg_label = pd.DataFrame(np.zeros(negnum))

    label = pd.concat([pos_label, neg_label], axis=0)
    label.index = list(range(posnum + negnum))
    dataset = pd.concat([pos_protein, neg_protein], axis=0)
    dataset.index = list(range(posnum + negnum))
    return dataset, label


def get_model(dropout_value=0.2):
    input_1 = Input(shape=(1164, ), name='Protein_a')
    protein_input1 = Dense(
        512,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proA_feature_1')(input_1)
    protein_input1 = Dropout(dropout_value)(protein_input1)
    protein_input1 = Dense(
        256,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proA_feature_2')(protein_input1)
    protein_input1 = Dropout(dropout_value)(protein_input1)
    protein_input1 = Dense(
        128,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proA_feature_3')(protein_input1)
    protein_input1 = Dropout(dropout_value)(protein_input1)
    input_2 = Input(shape=(1164, ), name='Protein_b')
    protein_input2 = Dense(
        512,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proB_feature_1')(input_2)
    protein_input2 = Dropout(dropout_value)(protein_input2)
    protein_input2 = Dense(
        256,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proB_feature_2')(protein_input2)
    protein_input2 = Dropout(dropout_value)(protein_input2)
    protein_input2 = Dense(
        128,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_proB_feature_3')(protein_input2)
    protein_input2 = Dropout(dropout_value)(protein_input2)
    merged_vector = merge(
        [protein_input1, protein_input2],
        mode='concat',
        concat_axis=1,
        name='merge_pro_A_B')
    output = Dense(
        128,
        activation='relu',
        kernel_initializer='glorot_normal',
        name='High_dim_feature_1')(merged_vector)
    outputs = Dense(2, activation='softmax', name='output')(output)
    model = Model(input=[input_1, input_2], output=outputs)
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(
        loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.summary()
    return model


def get_shuffle(dataset, label, random_state):
    #shuffle data
    np.random.seed(random_state)
    index = list(range(len(label)))
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset, label


def start_fit(dataset, label, title):
    #pre-processing
    dataset = np.array(dataset)
    label = np.array(label)
    dataset, label = get_shuffle(dataset, label, random_state=1)

    #split dataset to train set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, label, random_state=0)

    #normalization
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X1_train = X_train[:, 0:1164]
    X2_train = X_train[:, 1164:2328]
    X1_test = X_test[:, 0:1164]
    X2_test = X_test[:, 1164:2328]

    model = get_model(dropout_value=0.2)

    y_train = np_utils.to_categorical(y_train)

    #fit
    model.fit(
        [X1_train, X2_train],
        y_train,
        nb_epoch=30,
        batch_size=64,
        verbose=0, )

    #prediction probability
    y_probas = model.predict([X1_test, X2_test])
    y_test = np_utils.to_categorical(y_test)
    fpr, tpr, _ = roc_curve(y_test[:, 0], y_probas[:, 0])
    roc_auc = auc(fpr, tpr)

    draw_roc(y_test, y_probas)
    draw_pr(y_test, y_probas)

    y_class = categorical_probas_to_classes(y_probas)
    y_test = categorical_probas_to_classes(y_test)
    acc, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performace(
        len(y_class), y_class, y_test)
    print(title)
    print((
        'DeepPPI:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,roc_auc=%f'
        % (acc, precision, npv, sensitivity, specificity, mcc, roc_auc)))


#%%
#load yeast dataset
yeast_feature = 'dataset/Yeast/yeast_feature.csv'
yeast_protein = 'dataset/Yeast/yeast_protein.csv'
yeast_dataset, yeast_label = get_dataset(yeast_feature, yeast_protein, 5594,
                                         5594)
start_fit(yeast_dataset, yeast_label, "Test on yeast dataset")

#load pylori dataset
pylori_feature = 'dataset/pylori/pylori_feature.csv'
pylori_protein = 'dataset/pylori/pylori_protein.csv'
pylori_dataset, pylori_label = get_dataset(pylori_feature, pylori_protein,
                                           1458, 1458)
nan_rows = pylori_dataset[pylori_dataset.isnull().T.any().T]
row = [int(x) for x in nan_rows.index]
pylori_dataset = pylori_dataset.drop(row)
pylori_label = pylori_label.drop(row)
start_fit(pylori_dataset, pylori_label, "Test on yeast pylori")

#load human dataset
human_feature = 'dataset/Human/human_feature.csv'
human_protein = 'dataset/Human/human_protein.csv'
human_dataset, human_label = get_dataset(human_feature, human_protein, 3899,
                                         4262)
start_fit(human_dataset, human_label, "Test on yeast human")
