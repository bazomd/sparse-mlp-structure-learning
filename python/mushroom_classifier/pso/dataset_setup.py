import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection
import torch
from torch.autograd import Variable
import os
from common.dataset_model import Dataset


def setup(random_state):
    """
    Sets up the dataset with specific random state seed
    All categorical features are one-hot encoded
    """
    data = pd.read_csv(os.path.abspath('../python/mushroom_classifier/dataset/agaricus-lepiota.data'))
    y = pd.DataFrame(data['class'])
    data.drop('class', axis=1, inplace=True)
    feature_encoder = OneHotEncoder(drop='if_binary')
    x = feature_encoder.fit_transform(data).toarray()
    x = np.delete(np.array(x), 46, 1)  # delete the column of unknown value
    feature_names_raw = feature_encoder.get_feature_names()
    feature_names = fix_feature_names(feature_names_raw)
    feature_names = np.delete(np.array(feature_names), 46, 0)
    target_encoder = OneHotEncoder(drop='if_binary')
    y = target_encoder.fit_transform(y).toarray()
    y = y.reshape(1, -1)[0]
    target_names = np.array(['poisonous'])

    train_X, test_X, train_y, test_y = model_selection.train_test_split(x, y, test_size=0.2, random_state=random_state)
    train_X = Variable(torch.Tensor(train_X).float(), requires_grad=False)
    test_X = Variable(torch.Tensor(test_X).float(), requires_grad=False)
    train_y = Variable(torch.Tensor(train_y).float(), requires_grad=False)
    test_y = Variable(torch.Tensor(test_y).float(), requires_grad=False)
    return Dataset(train_X, train_y, test_X, test_y, data, feature_names, target_names)


def get_connected_features(connection_matrix):
    # deprecated
    connected_features = set()
    for i in range(connection_matrix.shape[1]):
        if np.count_nonzero(connection_matrix[:, i]) > 0:
            connected_features.add(i)
    return list(connected_features)


def fix_feature_names(feature_names_raw):
    """
    Properly formats feature names
    """
    feature_names_ = [i.replace('x0', 'cape-shape') for i in feature_names_raw[0:20]]
    feature_names_ = [i.replace('x1', 'cap-surface') for i in feature_names_[0:20]]
    feature_names_ = [i.replace('x2', 'cap-color') for i in feature_names_[0:20]] + feature_names_raw[20:].tolist()
    feature_names_ = [i.replace('x3', 'bruises') for i in feature_names_]
    feature_names_ = [i.replace('x4', 'odor') for i in feature_names_]
    feature_names_ = [i.replace('x5', 'gill-attachment') for i in feature_names_]
    feature_names_ = [i.replace('x6', 'gill-spacing') for i in feature_names_]
    feature_names_ = [i.replace('x7', 'gill-size') for i in feature_names_]
    feature_names_ = [i.replace('x8', 'gill-color') for i in feature_names_]
    feature_names_ = [i.replace('x9', 'stalk-shape') for i in feature_names_]
    feature_names_ = [i.replace('x10', 'stalk-root') for i in feature_names_]
    feature_names_ = [i.replace('x11', 'stalk-surface-above-ring') for i in feature_names_]
    feature_names_ = [i.replace('x12', 'stalk-surface-below-ring') for i in feature_names_]
    feature_names_ = [i.replace('x13', 'stalk-color-above-ring') for i in feature_names_]
    feature_names_ = [i.replace('x14', 'stalk-color-below-ring') for i in feature_names_]
    feature_names_ = [i.replace('x15', 'veil-type') for i in feature_names_]
    feature_names_ = [i.replace('x16', 'veil-color') for i in feature_names_]
    feature_names_ = [i.replace('x17', 'ring-number') for i in feature_names_]
    feature_names_ = [i.replace('x18', 'ring-type') for i in feature_names_]
    feature_names_ = [i.replace('x19', 'spore-print-color') for i in feature_names_]
    feature_names_ = [i.replace('x20', 'population') for i in feature_names_]
    feature_names_ = [i.replace('x21', 'habitat') for i in feature_names_]
    return feature_names_


