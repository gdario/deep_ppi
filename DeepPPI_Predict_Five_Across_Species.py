# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 18:05:54 2017

@author: sun
"""

import numpy as np
import pandas as pd
import h5py
from keras.utils import np_utils
from keras.models import load_model
from utils.tools import categorical_probas_to_classes


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


def get_across_species_dataset(feature_file, protein_file, posnum):
    feature = pd.read_csv(feature_file, index_col=0)
    protein = pd.read_csv(protein_file)
    protein = feature.loc[protein.id, :]
    P_protein_a = protein.iloc[0:posnum, :]
    P_protein_b = protein.iloc[posnum:posnum * 2, :]
    P_protein_a.index = list(range(posnum))
    P_protein_b.index = list(range(posnum))
    P_protein_a = np.array(P_protein_a)
    P_protein_b = np.array(P_protein_b)
    return P_protein_a, P_protein_b


def get_acc(protein_a, protein_b):
    #load DeepPPI model
    model = load_model('model/DeepPPI_all_train.h5')

    # read method open file
    file = h5py.File('model/Scaler_all_train.h5', 'r')
    mean = file['mean'][:]
    std = file['std'][:]
    file.close()

    protein_a = (protein_a - mean[0:1164]) / std[0:1164]
    protein_b = (protein_b - mean[1164:2328]) / std[1164:2328]

    #use DeepPPI prediction
    y_probas = model.predict([protein_a, protein_b])
    y_class = list(categorical_probas_to_classes(y_probas))
    acc = 1.0 * y_class.count(1) / len(y_class)

    return acc


#%%
#load dataset
ecoli_feature = 'dataset/cross_species_dataset/ecoli_feature.csv'
ecoli_protein = 'dataset/cross_species_dataset/ecoli_protein.csv'
protein_a, protein_b = get_across_species_dataset(ecoli_feature, ecoli_protein,
                                                  6954)
ecoli_acc = get_acc(protein_a, protein_b)

celeg_feature = 'dataset/cross_species_dataset/celeg_feature.csv'
celeg_protein = 'dataset/cross_species_dataset/celeg_protein.csv'
protein_a, protein_b = get_across_species_dataset(celeg_feature, celeg_protein,
                                                  4013)
celeg_acc = get_acc(protein_a, protein_b)

hsapi_feature = 'dataset/cross_species_dataset/hsapi_feature.csv'
hsapi_protein = 'dataset/cross_species_dataset/hsapi_protein.csv'
protein_a, protein_b = get_across_species_dataset(hsapi_feature, hsapi_protein,
                                                  1412)
hsapi_acc = get_acc(protein_a, protein_b)

hpylo_feature = 'dataset/cross_species_dataset/hpylo_feature.csv'
hpylo_protein = 'dataset/cross_species_dataset/hpylo_protein.csv'
protein_a, protein_b = get_across_species_dataset(hpylo_feature, hpylo_protein,
                                                  1420)
hpylo_acc = get_acc(protein_a, protein_b)

mmusc_feature = 'dataset/cross_species_dataset/mmusc_feature.csv'
mmusc_protein = 'dataset/cross_species_dataset/mmusc_protein.csv'
protein_a, protein_b = get_across_species_dataset(mmusc_feature, mmusc_protein,
                                                  313)
mmusc_acc = get_acc(protein_a, protein_b)

print(("""E.coli Acc:%.2f
C.elegans Acc:%.2f
H.sapiens Acc:%.2f
H.pylori Acc:%.2f
M.musculus Acc:%.2f""" % (ecoli_acc * 100, celeg_acc * 100, hsapi_acc * 100,
                          hpylo_acc * 100, mmusc_acc * 100)))
