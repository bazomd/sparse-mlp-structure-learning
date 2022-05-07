import numpy as np
from sklearn import datasets
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection
import torch
from torch.autograd import Variable

from common.dataset_model import Dataset


def setup(random_state, bin_size=10, discretization_strategy='quantile'):
    raw_dataset = datasets.load_iris()
    # Calculate the number of bins for a feature (~ one for 10 samples)
    X_ = raw_dataset.data
    Y_ = raw_dataset.target

    one_hot_encoder = OneHotEncoder(sparse=False)
    sl_discretizer = KBinsDiscretizer(n_bins=cal_num_of_bins(X_[:, 0], bin_size), encode='onehot-dense',
                                      strategy=discretization_strategy)
    sw_discretizer = KBinsDiscretizer(n_bins=cal_num_of_bins(X_[:, 1], bin_size), encode='onehot-dense',
                                      strategy=discretization_strategy)
    pl_discretizer = KBinsDiscretizer(n_bins=cal_num_of_bins(X_[:, 2], bin_size), encode='onehot-dense',
                                      strategy=discretization_strategy)
    pw_discretizer = KBinsDiscretizer(n_bins=cal_num_of_bins(X_[:, 3], bin_size), encode='onehot-dense',
                                      strategy=discretization_strategy)

    sl = sl_discretizer.fit_transform(np.array(X_[:, 0]).reshape(-1, 1))
    sw = sw_discretizer.fit_transform(np.array(X_[:, 1]).reshape(-1, 1))
    pl = pl_discretizer.fit_transform(np.array(X_[:, 2]).reshape(-1, 1))
    pw = pw_discretizer.fit_transform(np.array(X_[:, 3]).reshape(-1, 1))
    x = np.concatenate((sl, sw, pl, pw), axis=1)

    print('Bin edges:')
    print('sl: ', sl_discretizer.bin_edges_.tolist())
    print('sw: ', sw_discretizer.bin_edges_.tolist())
    print('pl: ', pl_discretizer.bin_edges_.tolist())
    print('pw: ', pw_discretizer.bin_edges_.tolist())

    feature_names = ['sl: ' + str(sl_discretizer.bin_edges_.tolist()[0][0:2]),
                     'sl: ' + str(sl_discretizer.bin_edges_.tolist()[0][1:3]),
                     'sl: ' + str(sl_discretizer.bin_edges_.tolist()[0][2:4]),
                     'sl: ' + str(sl_discretizer.bin_edges_.tolist()[0][3:5]),
                     'sw: ' + str(sw_discretizer.bin_edges_.tolist()[0][0:2]),
                     'sw: ' + str(sw_discretizer.bin_edges_.tolist()[0][1:3]),
                     'pl: ' + str(pl_discretizer.bin_edges_.tolist()[0][0:2]),
                     'pl: ' + str(pl_discretizer.bin_edges_.tolist()[0][1:3]),
                     'pl: ' + str(pl_discretizer.bin_edges_.tolist()[0][2:4]),
                     'pl: ' + str(pl_discretizer.bin_edges_.tolist()[0][3:5]),
                     'pw: ' + str(pw_discretizer.bin_edges_.tolist()[0][0:2]),
                     'pw: ' + str(pw_discretizer.bin_edges_.tolist()[0][1:3])]

    reshaped = Y_.reshape(len(Y_), 1)
    # y = one_hot_encoder.fit_transform(reshaped)

    train_X, test_X, train_y, test_y = model_selection.train_test_split(x, Y_, test_size=0.2, random_state=random_state)

    # wrap up with Variable in pytorch
    train_X = Variable(torch.Tensor(train_X).float(), requires_grad=False)
    test_X = Variable(torch.Tensor(test_X).float(), requires_grad=False)
    train_y = Variable(torch.Tensor(train_y).long(), requires_grad=False)
    test_y = Variable(torch.Tensor(test_y).long(), requires_grad=False)
    dataset = Dataset(train_X, train_y, test_X, test_y, raw_dataset, feature_names, raw_dataset.target_names)
    return dataset


def cal_num_of_bins(arr, bin_size):
    s = len(set(arr))
    return int(round(s / bin_size))
