import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn import model_selection
import numpy as np
import torch
from torch.autograd import Variable

from common.dataset_model import Dataset


def setup(random_state):
    discretization_strategy = 'quantile'
    data = pd.read_csv(os.path.abspath('../python/income_classifiers/dataset/adult.csv'))
    data.drop('education.num', axis=1, inplace=True)
    data[data == '?'] = np.nan
    data = data.dropna(axis=0)
    one_hot_encoder = OneHotEncoder(drop='if_binary', sparse=False)

    y = pd.DataFrame(data['income'])
    data.drop('income', axis=1, inplace=True)
    target_encoder = OneHotEncoder(drop='if_binary')
    y = target_encoder.fit_transform(y).toarray()
    y = y.reshape(1, -1)[0]
    target_names = np.array(['income _>50K'])

    # number of bins was taken experimentally
    age_discretizer = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
    f_age = age_discretizer.fit_transform(pd.DataFrame(data['age']))
    f_age_names = ['age: ' + str(age_discretizer.bin_edges_.tolist()[0][i:i + 2]) for i in
                   range(len(age_discretizer.bin_edges_[0]) - 1)]

    f_workclass = one_hot_encoder.fit_transform(pd.DataFrame(data['workclass']))
    f_workclass = np.delete(f_workclass, 0, 1)  # delete column with '?' as a value
    f_workclass_names = one_hot_encoder.get_feature_names()
    f_workclass_names = np.delete(f_workclass_names, 0, 0)
    f_workclass_names = [i.replace('x0', 'workclass') for i in f_workclass_names]

    fnlwgt_discretizer = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
    f_fnlwgt = fnlwgt_discretizer.fit_transform(pd.DataFrame(data['fnlwgt']))
    f_fnlwgt_names = ['fnlwgt: ' + str(fnlwgt_discretizer.bin_edges_.tolist()[0][i:i + 2]) for i in
                      range(len(fnlwgt_discretizer.bin_edges_[0]) - 1)]

    f_education = one_hot_encoder.fit_transform(pd.DataFrame(data['education']))
    f_education_names = one_hot_encoder.get_feature_names()
    f_education_names = [i.replace('x0', 'education') for i in f_education_names]

    f_marital_status = one_hot_encoder.fit_transform(pd.DataFrame(data['marital.status']))
    f_marital_status_names = one_hot_encoder.get_feature_names()
    f_marital_status_names = [i.replace('x0', 'marital.status') for i in f_marital_status_names]

    f_occupation = one_hot_encoder.fit_transform(pd.DataFrame(data['occupation']))
    f_occupation = np.delete(f_occupation, 0, 1)
    f_occupation_names = one_hot_encoder.get_feature_names()
    f_occupation_names = np.delete(f_occupation_names, 0, 0)
    f_occupation_names = [i.replace('x0', 'occupation') for i in f_occupation_names]

    f_relationship = one_hot_encoder.fit_transform(pd.DataFrame(data['relationship']))
    f_relationship_names = one_hot_encoder.get_feature_names()
    f_relationship_names = [i.replace('x0', 'relationship') for i in f_relationship_names]

    f_race = one_hot_encoder.fit_transform(pd.DataFrame(data['race']))
    f_race_names = one_hot_encoder.get_feature_names()
    f_race_names = [i.replace('x0', 'race') for i in f_race_names]

    f_sex = one_hot_encoder.fit_transform(pd.DataFrame(data['sex']))
    f_sex_names = one_hot_encoder.get_feature_names()
    f_sex_names = [i.replace('x0', 'sex') for i in f_sex_names]

    capital_gain_discretizer = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')
    f_capital_gain = capital_gain_discretizer.fit_transform(pd.DataFrame(data['capital.gain']))
    f_capital_gain_names = ['capital.gain: ' + str(capital_gain_discretizer.bin_edges_.tolist()[0][i:i + 2]) for i in
                            range(len(capital_gain_discretizer.bin_edges_[0]) - 1)]

    capital_loss_discretizer = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')
    f_capital_loss = capital_loss_discretizer.fit_transform(pd.DataFrame(data['capital.loss']))
    f_capital_loss_names = ['capital.loss: ' + str(capital_loss_discretizer.bin_edges_.tolist()[0][i:i + 2]) for i in
                            range(len(capital_loss_discretizer.bin_edges_[0]) - 1)]

    hours_per_week_discretizer = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')
    f_hours_per_week = hours_per_week_discretizer.fit_transform(pd.DataFrame(data['hours.per.week']))
    f_hours_per_week_names = ['hours.per.week: ' + str(hours_per_week_discretizer.bin_edges_.tolist()[0][i:i + 2]) for i
                              in range(len(hours_per_week_discretizer.bin_edges_[0]) - 1)]

    f_native_country = one_hot_encoder.fit_transform(pd.DataFrame(data['native.country']))
    f_native_country = np.delete(f_native_country, 0, 1)
    f_native_country_names = one_hot_encoder.get_feature_names()
    f_native_country_names = np.delete(f_native_country_names, 0, 0)
    f_native_country_names = [i.replace('x0', 'native.country') for i in f_native_country_names]

    x = np.concatenate((f_age, f_workclass, f_fnlwgt, f_education, f_marital_status, f_occupation,
                        f_relationship, f_race, f_sex, f_capital_gain, f_capital_loss, f_hours_per_week,
                        f_native_country), axis=1)

    feature_names = f_age_names + f_workclass_names + f_fnlwgt_names + f_education_names + f_marital_status_names \
                    + f_occupation_names + f_relationship_names + f_race_names + f_sex_names + f_capital_gain_names \
                    + f_capital_loss_names + f_hours_per_week_names + f_native_country_names

    train_X, test_X, train_y, test_y = model_selection.train_test_split(x, y, test_size=0.2, random_state=random_state)
    train_X = Variable(torch.Tensor(train_X).float(), requires_grad=False)
    test_X = Variable(torch.Tensor(test_X).float(), requires_grad=False)
    train_y = Variable(torch.Tensor(train_y).float(), requires_grad=False)
    test_y = Variable(torch.Tensor(test_y).float(), requires_grad=False)
    dataset = Dataset(train_X, train_y, test_X, test_y, data, feature_names, target_names)
    return dataset


def cal_num_of_bins(column):
    x = column.array.to_numpy()
    n = len(x)
    # Sturges binning
    k = np.ceil(np.log2(n)) + 1
    return k


# setup()
