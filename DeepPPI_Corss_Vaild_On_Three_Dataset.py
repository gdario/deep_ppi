# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 10:46:22 2017

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from utils.tools import calculate_performace, categorical_probas_to_classes


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


def get_shuffle(dataset, label, random_state):
    #shuffle data
    np.random.seed(random_state)
    index = list(range(len(label)))
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset, label


def start_fit(dataset, label, title):
    dataset = np.array(dataset)
    label = np.array(label)
    dataset, label = get_shuffle(dataset, label, random_state=1)

    #normalization
    scaler = StandardScaler().fit(dataset)
    dataset = scaler.transform(dataset)

    X1_train = dataset[:, 0:1164]
    X2_train = dataset[:, 1164:2328]
    label = label.reshape(
        len(label), )
    y_train = np_utils.to_categorical(label)

    # define 5-fold cross validation test harness
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    cvscores = []

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lw = 2
    i = 0

    for train, test in skf.split(dataset, label):
        model = get_model(dropout_value=0.2)
        model.fit(
            [X1_train[train], X2_train[train]],
            y_train[train],
            epochs=30,
            batch_size=64,
            verbose=0, )
        #prediction probability
        y_probas = model.predict([X1_train[test], X2_train[test]])

        fpr, tpr, _ = roc_curve(y_train[test][:, 0], y_probas[:, 0])
        roc_auc = auc(fpr, tpr)

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        plt.plot(
            fpr,
            tpr,
            lw=lw,
            color=plt.cm.Set1(i / 10.),
            label='ROC fold %d (area = %0.2f%%)' % (i, (roc_auc * 100)))

        i += 1

        y_class = categorical_probas_to_classes(y_probas)
        y_test = categorical_probas_to_classes(y_train[test])

        acc, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performace(
            len(y_class), y_class, y_test)
        cvscores.append(
            [acc, precision, npv, sensitivity, specificity, mcc, roc_auc])

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

    mean_tpr /= skf.get_n_splits(dataset, label)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color='g',
        linestyle='--',
        label='Mean ROC (area = %0.2f%%)' % (mean_auc * 100),
        lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print(title)
    scores = np.array(cvscores)
    print(("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0] * 100,
                                        np.std(scores, axis=0)[0] * 100)))
    print(
        ("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1] * 100,
                                            np.std(scores, axis=0)[1] * 100)))
    print(("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2] * 100,
                                        np.std(scores, axis=0)[2] * 100)))
    print(("sensitivity=%.2f%% (+/- %.2f%%)" %
           (np.mean(scores, axis=0)[3] * 100,
            np.std(scores, axis=0)[3] * 100)))
    print(("specificity=%.2f%% (+/- %.2f%%)" %
           (np.mean(scores, axis=0)[4] * 100,
            np.std(scores, axis=0)[4] * 100)))
    print(("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5] * 100,
                                        np.std(scores, axis=0)[5] * 100)))
    print(("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6] * 100,
                                            np.std(scores, axis=0)[6] * 100)))


#%%
#load dataset
yeast_feature = 'dataset/Yeast/yeast_feature.csv'
yeast_protein = 'dataset/Yeast/yeast_protein.csv'
yeast_dataset, yeast_label = get_dataset(yeast_feature, yeast_protein, 5594,
                                         5594)
start_fit(yeast_dataset, yeast_label, "Five corss vaild on yeast dataset")

pylori_feature = 'dataset/pylori/pylori_feature.csv'
pylori_protein = 'dataset/pylori/pylori_protein.csv'
pylori_dataset, pylori_label = get_dataset(pylori_feature, pylori_protein,
                                           1458, 1458)
nan_rows = pylori_dataset[pylori_dataset.isnull().T.any().T]
row = [int(x) for x in nan_rows.index]
pylori_dataset = pylori_dataset.drop(row)
pylori_label = pylori_label.drop(row)
start_fit(pylori_dataset, pylori_label, "Five corss vaild yeast pylori")

human_feature = 'dataset/Human/human_feature.csv'
human_protein = 'dataset/Human/human_protein.csv'
human_dataset, human_label = get_dataset(human_feature, human_protein, 3899,
                                         4262)
start_fit(human_dataset, human_label, "Five corss vaild yeast human")
