#!/usr/bin/env python
# coding: utf-8
# author: Bo Tang

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer

def loadData(dataname):
    """
    load training and testing data from different dataset
    """
    # balance-scale
    if dataname == 'balance-scale':
        x, y = loadBalanceScale()
        return x, y
    if dataname == 'banknote_train':
        x, y = loadBanknote()
        return x, y
    # breast-cancer
    if dataname == 'breast-cancer':
        x, y, feature_mappings, label_mapping = loadBreastCancer()
        return x, y, feature_mappings, label_mapping
    # dataset
    if dataname == 'dataset':
        x, y, feature_mappings, label_mapping = loadDataset()
        return x, y, feature_mappings, label_mapping
    # dermatology
    if dataname == 'dermatology':
        x, y, feature_mappings, label_mapping = loadDermatology()
        return x, y, feature_mappings, label_mapping
    # car-evaluation
    if dataname == 'car-evaluation':
        x, y, feature_mappings, label_mapping = loadCarEvaluation()
        return x, y, feature_mappings, label_mapping
    # hayes-roth
    if dataname == 'hayes-roth':
        x, y = loadHayesRoth()
        return x, y
    # house-votes-84
    if dataname == 'house-votes-84':
        x, y = loadHouseVotes84()
        return x, y
    # soybean-small
    if dataname == 'soybean-small':
        x, y = loadSoybean()
        return x, y
    # spect
    if dataname == 'spect':
        x, y = loadSpect()
        return x, y
    # tic-tac-toe
    if dataname == 'tic-tac-toe':
        x, y = loadTicTacToe()
        return x, y
    # monks
    if dataname[:5] == 'monks':
        x, y = loadMonks(dataname)
        return x, y
    raise NameError('No dataset "{}".'.format(dataname))

def oneHot(x):
    """
    one-hot encoding
    """
    x_enc = np.zeros((x.shape[0], 0))
    for j in range(x.shape[1]):
        lb = LabelBinarizer()
        lb.fit(np.unique(x[:,j]))
        x_enc = np.concatenate((x_enc, lb.transform(x[:,j])), axis=1)
    return x_enc

def loadBalanceScale():
    """
    load balance-scale dataset
    """
    df = pd.read_csv('./data/balance-scale/balance-scale.data', header=None, delimiter=',')
    x, y = df[[1,2,3,4]], df[0]
    y = pd.factorize(y)
    return np.array(x), np.array(y, dtype=object)[0]

def loadBanknote():
    """
    Load banknote dataset
    """
    df = pd.read_csv('./data/banknote_train/banknote_train.csv', header=None, delimiter=',')
    x, y = df.iloc[:, :-1], df.iloc[:, -1]
    return np.array(x), np.array(y)


def loadBreastCancer():
    """
    load breast-cancer dataset
    """
    # Lire le fichier CSV
    df = pd.read_csv('./data/breast-cancer/breast-cancer.data', header=None, delimiter=',')

    # Supprimer les lignes contenant des valeurs manquantes
    df = df.replace('?', np.nan).dropna()

    # Encoder les caractéristiques et les étiquettes
    feature_columns = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    label_column = 0

    feature_mappings = {}
    for col in feature_columns:
        df[col], feature_mappings[col] = pd.factorize(df[col], sort=True)

    # Encoder les étiquettes
    df[label_column], label_mapping = pd.factorize(df[label_column], sort=True)

    # Séparer les caractéristiques (X) et les étiquettes (y)
    x = df[feature_columns].values
    y = df[label_column].values

    return x, y, feature_mappings, label_mapping
def loadDataset():
    """
    Load dataset from dataset.data file.
    """
    # Lire le fichier CSV
    df = pd.read_csv('./data/dataset/dataset.data', header=None, delimiter=',')

    # Renommer les colonnes
    df.columns = ['feature', 'class']

    # Encoder les caractéristiques et les étiquettes
    feature_columns = ['feature']
    label_column = 'class'

    feature_mappings = {}
    for col in feature_columns:
        df[col], feature_mappings[col] = pd.factorize(df[col], sort=True)

    # Encoder les étiquettes
    df[label_column], label_mapping = pd.factorize(df[label_column], sort=True)

    # Séparer les caractéristiques (X) et les étiquettes (y)
    x = df[feature_columns].values
    y = df[label_column].values

    return x, y, feature_mappings, label_mapping

def loadDermatology():
    """
    Load dermatology dataset
    """
    # Lire le fichier CSV
    df = pd.read_csv('./data/dermatology/dermatology.data', header=None, delimiter=',')

    # Supprimer les lignes contenant des valeurs manquantes
    df = df.replace('?', np.nan).dropna()

    # Définir les colonnes des caractéristiques et des étiquettes
    feature_columns = list(range(34))  # Les 34 premières colonnes sont des caractéristiques
    label_column = 34  # La dernière colonne est l'étiquette

    # Encoder les caractéristiques et les étiquettes
    feature_mappings = {}
    for col in feature_columns:
        df[col], feature_mappings[col] = pd.factorize(df[col], sort=True)

    # Encoder les étiquettes
    df[label_column], label_mapping = pd.factorize(df[label_column], sort=True)

    # Séparer les caractéristiques (X) et les étiquettes (y)
    x = df[feature_columns].values
    y = df[label_column].values

    return x, y, feature_mappings, label_mapping

def loadCarEvaluation():
    """
    load car-evaluation dataset
    """
    # Lire le fichier CSV
    df = pd.read_csv('./data/car-evaluation/car.data', header=None, delimiter=',')

    # Encoder les caractéristiques
    feature_columns = [0, 1, 2, 3, 4, 5]
    label_column = 6

    feature_mappings = {}
    for col in feature_columns:
        df[col], feature_mappings[col] = pd.factorize(df[col], sort=True)

    # Encoder les étiquettes
    df[label_column], label_mapping = pd.factorize(df[label_column], sort=True)

    # Séparer les caractéristiques (X) et les étiquettes (y)
    x = df[feature_columns].values
    y = df[label_column].values

    return x, y, feature_mappings, label_mapping

def loadHayesRoth():
    """
    load hayes-roth dataset
    """
    df_train = pd.read_csv('./data/hayes-roth/hayes-roth.data', header=None, delimiter=',')
    df_test = pd.read_csv('./data/hayes-roth/hayes-roth.test', header=None, delimiter=',')
    x_train, y_train = df_train[[1,2,3,4]], df_train[5]
    x_test, y_test = df_test[[0,1,2,3]], df_test[4]
    x, y = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
    x = pd.DataFrame(x)
    x1, x2 = np.array(x[[0,3]]), np.array(x[[1,2]])
    x1 = oneHot(x1)
    x = np.concatenate((x1, x2), axis=1)
    return x, y

def loadHouseVotes84():
    """
    load house-votes-84 dataset
    """
    df = pd.read_csv('./data/house-votes-84/house-votes-84.data', header=None, delimiter=',')
    for i in range(1,17):
        df = df[df[i] != '?']
    df = df.apply(lambda x: pd.factorize(x)[0])
    x, y = df[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]], df[0]
    return np.array(x), np.array(y)

def loadSoybean():
    """
    load soybean dataset
    """
    df = pd.read_csv('./data/soybean-small/soybean-small.data', header=None, delimiter=',')
    for i in range(35):
        df = df[df[i] != '?']
    x, y = df[range(35)], df[35]
    y = pd.factorize(y)
    x = pd.DataFrame(x)
    x1 = np.array(x[[0,5,8,12,13,14,17,20,21,23,25,27,28,34]])
    x2 = np.array(x[[1,2,3,4,6,7,9,10,11,15,16,18,19,22,24,26,29,30,31,32,33]])
    x1 = oneHot(x1)
    x = np.concatenate((x1, x2), axis=1)
    return np.array(x), np.array(y, dtype=object)[0]

def loadSpect():
    """
    load spect dataset
    """
    df_train = pd.read_csv('./data/spect/spect.train', header=None, delimiter=',')
    df_test = pd.read_csv('./data/spect/spect.test', header=None, delimiter=',')
    x_train, y_train = df_train[range(1,23)], df_train[0]
    x_test, y_test = df_test[range(1,23)], df_test[0]
    return np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)

def loadTicTacToe():
    """
    load tic-tac-toe dataset
    """
    df = pd.read_csv('./data/tic-tac-toe/tic-tac-toe.data', header=None, delimiter=',')
    x, y = df[[0,1,2,3,4,5,6,7]], df[9]
    x = oneHot(np.array(x))
    y = pd.factorize(y)
    return x, np.array(y, dtype=object)[0]

def loadMonks(dataname):
    """
    load Monks dataset
    """
    _, index = dataname.split('-')
    if index not in ['1', '2', '3']:
        raise AssertionError('No dataset ' + dataname)
    df_train = pd.read_csv('./data/monks/monks-{}.train'.format(index), header=None, delimiter=' ')
    df_test = pd.read_csv('./data/monks/monks-{}.test'.format(index), header=None, delimiter=' ')
    x_train, y_train = df_train[[2,3,4,5,6,7]], df_train[1]
    x_test, y_test = df_test[[2,3,4,5,6,7]], df_test[1]
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    x = oneHot(x)
    return x, y
