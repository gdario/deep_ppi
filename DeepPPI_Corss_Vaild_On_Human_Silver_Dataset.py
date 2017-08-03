# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:39:46 2017

@author: sun
"""
import numpy as np
import pandas as pd

from keras.layers import Dense, Input, Dropout
from keras.layers.merge import concatenate
from keras.optimizers import SGD, RMSprop
from keras.models import Model
from keras.regularizers import l2

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils


def get_sep_model():
    input_1 = Input(shape=(4, ), name='Protein_a')
    protein_input1 = Dense(
        512,
        activation='relu',
        kernel_initializer='he_normal',
        name='High_dim_proA_feature_1',
        kernel_regularizer=l2(0.01))(input_1)
    protein_input1 = Dropout(0.2)(protein_input1)
    protein_input1 = Dense(
        256,
        activation='relu',
        kernel_initializer='he_normal',
        name='High_dim_proA_feature_2',
        kernel_regularizer=l2(0.01))(protein_input1)
    protein_input1 = Dropout(0.2)(protein_input1)
    protein_input1 = Dense(
        128,
        activation='relu',
        kernel_initializer='he_normal',
        name='High_dim_proA_feature_3',
        kernel_regularizer=l2(0.01))(protein_input1)
    protein_input1 = Dropout(0.2)(protein_input1)
    input_2 = Input(shape=(4, ), name='Protein_b')
    protein_input2 = Dense(
        512,
        activation='relu',
        kernel_initializer='he_normal',
        name='High_dim_proB_feature_1',
        kernel_regularizer=l2(0.01))(input_2)
    protein_input2 = Dropout(0.2)(protein_input2)
    protein_input2 = Dense(
        256,
        activation='relu',
        kernel_initializer='he_normal',
        name='High_dim_proB_feature_2',
        kernel_regularizer=l2(0.01))(protein_input2)
    protein_input2 = Dropout(0.2)(protein_input2)
    protein_input2 = Dense(
        128,
        activation='relu',
        kernel_initializer='he_normal',
        name='High_dim_proB_feature_3',
        kernel_regularizer=l2(0.01))(protein_input2)
    protein_input2 = Dropout(0.2)(protein_input2)
    merged_vector = concatenate([protein_input1, protein_input2], axis=1)
    output = Dense(
        128,
        activation='relu',
        kernel_initializer='he_uniform',
        name='High_dim_feature_1')(merged_vector)
    outputs = Dense(2, activation='softmax', name='output')(output)
    model = Model(inputs=[input_1, input_2], outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(0.0001),
        metrics=['accuracy'])
    return model


yeast_gold_protein = pd.read_csv(
    'dataset/Human/human_silver.tab', sep='	').iloc[2:, 2:]

X = yeast_gold_protein.iloc[:, :4].values.astype("float")
y = yeast_gold_protein.iloc[:, 4].values.astype("int")

skf = StratifiedKFold(n_splits=10, shuffle=True)
sepscores = []
for train, test in skf.split(X, y):
    model = get_sep_model()
    y_train = utils.to_categorical(y[train])
    hist = model.fit(
        [X[train], X[train]],
        y_train,
        epochs=100,
        #validation_split=0.1,
        batch_size=32,
        verbose=0)
    #utils.plothistory(hist)
    #prediction probability
    y_score = model.predict([X[test], X[test]])
    y_test = utils.to_categorical(y[test])
    fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)

    y_class = utils.categorical_probas_to_classes(y_score)
    y_test_tmp = y[test]
    acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(
        len(y_class), y_class, y_test_tmp)
    sepscores.append(
        [acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])
    print((
        'DeepPPI-sep:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
        % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc)))
scores = np.array(sepscores)
print(("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0] * 100,
                                    np.std(scores, axis=0)[0] * 100)))
print(("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1] * 100,
                                          np.std(scores, axis=0)[1] * 100)))
print(("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2] * 100,
                                    np.std(scores, axis=0)[2] * 100)))
print(("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3] * 100,
                                            np.std(scores, axis=0)[3] * 100)))
print(("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4] * 100,
                                            np.std(scores, axis=0)[4] * 100)))
print(("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5] * 100,
                                    np.std(scores, axis=0)[5] * 100)))
print(("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6] * 100,
                                   np.std(scores, axis=0)[6] * 100)))
print(("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7] * 100,
                                        np.std(scores, axis=0)[7] * 100)))
