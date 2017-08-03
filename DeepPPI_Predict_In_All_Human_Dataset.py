# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:36:43 2017

@author: sun
"""

import numpy as np
import pandas as pd

from keras.layers import Dense, merge, Input, Dropout
from keras.optimizers import SGD, RMSprop

from keras.models import Model
from keras.regularizers import l2

from sklearn.metrics import roc_curve, auc
import utils.tools as utils


def get_sep_model():
    input_1 = Input(shape=(4, ), name='Protein_a')
    protein_input1 = Dense(
        512,
        activation='relu',
        init='he_normal',
        name='High_dim_proA_feature_1',
        W_regularizer=l2(0.01))(input_1)
    protein_input1 = Dropout(0.2)(protein_input1)
    protein_input1 = Dense(
        256,
        activation='relu',
        init='he_normal',
        name='High_dim_proA_feature_2',
        W_regularizer=l2(0.01))(protein_input1)
    protein_input1 = Dropout(0.2)(protein_input1)
    protein_input1 = Dense(
        128,
        activation='relu',
        init='he_normal',
        name='High_dim_proA_feature_3',
        W_regularizer=l2(0.01))(protein_input1)
    protein_input1 = Dropout(0.2)(protein_input1)
    input_2 = Input(shape=(4, ), name='Protein_b')
    protein_input2 = Dense(
        512,
        activation='relu',
        init='he_normal',
        name='High_dim_proB_feature_1',
        W_regularizer=l2(0.01))(input_2)
    protein_input2 = Dropout(0.2)(protein_input2)
    protein_input2 = Dense(
        256,
        activation='relu',
        init='he_normal',
        name='High_dim_proB_feature_2',
        W_regularizer=l2(0.01))(protein_input2)
    protein_input2 = Dropout(0.2)(protein_input2)
    protein_input2 = Dense(
        128,
        activation='relu',
        init='he_normal',
        name='High_dim_proB_feature_3',
        W_regularizer=l2(0.01))(protein_input2)
    protein_input2 = Dropout(0.2)(protein_input2)
    merged_vector = merge(
        [protein_input1, protein_input2],
        mode='concat',
        concat_axis=1,
        name='merge_pro_A_B')
    output = Dense(
        128, activation='relu', init='he_normal',
        name='High_dim_feature_1')(merged_vector)
    outputs = Dense(2, activation='softmax', name='output')(output)
    model = Model(input=[input_1, input_2], output=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'])
    return model


def train_sep_model(protein):
    X = protein.iloc[:, :4].values.astype("float")
    y = protein.iloc[:, 4].values.astype("int")
    #shuffle data
    np.random.seed(1)
    index = [i for i in range(len(y))]
    np.random.shuffle(index)
    X = X[index]
    y = y[index]

    model = get_sep_model()
    y_train = utils.to_categorical(y)
    model.fit(
        [X, X],
        y_train,
        nb_epoch=100,
        #validation_split=0.1,
        batch_size=32,
        verbose=0)
    return model


def pred_sep_model(model, X_test, y_test):
    y_score = model.predict([X_test, X_test])
    y_test_tmp = utils.to_categorical(y_test)
    fpr, tpr, _ = roc_curve(y_test_tmp[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)
    y_class = utils.categorical_probas_to_classes(y_score)
    y_test_tmp = y_test
    acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(
        len(y_class), y_class, y_test_tmp)
    print((
        'DeepPPI-sep:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
        % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc)))


human_gold_protein = pd.read_csv(
    'dataset/Human/human_gold.tab', sep='	').iloc[2:, 2:]
human_silver_protein = pd.read_csv(
    'dataset/Human/human_silver.tab', sep='	').iloc[2:, 2:]

gold_sep_model = train_sep_model(human_gold_protein)

silver_sep_model = train_sep_model(human_silver_protein)

test_set = pd.read_csv('dataset/Human/human_all.tab', sep='	').iloc[2:, 2:]
X_test = test_set.iloc[:, :4].values.astype("float")
y_test = test_set.iloc[:, 4].values.astype("int")
print('train on human gold test human all')
pred_sep_model(gold_sep_model, X_test, y_test)

print('train on human silver test human all')
pred_sep_model(silver_sep_model, X_test, y_test)
