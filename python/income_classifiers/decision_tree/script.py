import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main_script():
    max_depth = 12

    dataset = pd.read_csv('../python/income_classifiers/dataset/adult.csv')
    dataset.drop('education.num', axis=1, inplace=True)
    dataset[dataset == '?'] = np.nan
    dataset = dataset.dropna(axis=0)

    y = pd.DataFrame(dataset['income'])
    y = pd.get_dummies(y)
    dataset.drop('income', axis=1, inplace=True)

    age = pd.DataFrame(dataset['age'])
    workclass = pd.get_dummies(pd.DataFrame(dataset['workclass']))
    X = pd.DataFrame.join(age, workclass)
    fnlwgt = pd.DataFrame(dataset['fnlwgt'])
    X = pd.DataFrame.join(X, fnlwgt)
    education = pd.get_dummies(pd.DataFrame(dataset['education']))
    X = pd.DataFrame.join(X, education)
    marital_status = pd.get_dummies(pd.DataFrame(dataset['marital.status']))
    X = pd.DataFrame.join(X, marital_status)
    occupation = pd.get_dummies(pd.DataFrame(dataset['occupation']))
    X = pd.DataFrame.join(X, occupation)
    relationship = pd.get_dummies(pd.DataFrame(dataset['relationship']))
    X = pd.DataFrame.join(X, relationship)
    race = pd.get_dummies(pd.DataFrame(dataset['race']))
    X = pd.DataFrame.join(X, race)
    sex = pd.get_dummies(pd.DataFrame(dataset['sex']))
    X = pd.DataFrame.join(X, sex)
    capital_gain = pd.DataFrame(dataset['capital.gain'])
    X = pd.DataFrame.join(X, capital_gain)
    capital_loss = pd.DataFrame(dataset['capital.loss'])
    X = pd.DataFrame.join(X, capital_loss)
    hours_per_week = pd.DataFrame(dataset['hours.per.week'])
    X = pd.DataFrame.join(X, hours_per_week)
    native_country = pd.get_dummies(pd.DataFrame(dataset['native.country']))
    X = pd.DataFrame.join(X, native_country)

    test_accuracies = []
    test_recalls = []
    test_precisions = []
    test_f_1_scores = []

    train_accuracies = []
    train_recalls = []
    train_precisions = []
    train_f_1_scores = []

    for i in range(10):
        random_state = i
        np.random.seed(random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        model = tree.DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X_train, y_train)
        train_y_hat = model.predict(X_train)
        test_y_hat = model.predict(X_test)

        train_accuracies.append(accuracy_score(y_train, train_y_hat))
        train_precisions.append(precision_score(y_train, train_y_hat, average='micro'))
        train_recalls.append(recall_score(y_train, train_y_hat, average='micro'))
        train_f_1_scores.append(f1_score(y_train, train_y_hat, average='micro'))

        test_accuracies.append(accuracy_score(y_test, test_y_hat))
        test_precisions.append(precision_score(y_test, test_y_hat, average='micro'))
        test_recalls.append(recall_score(y_test, test_y_hat, average='micro'))
        test_f_1_scores.append(f1_score(y_test, test_y_hat, average='micro'))

    print('Average accuracy on training subset: ', np.mean(train_accuracies))
    print('Average recall on training subset: ', np.mean(train_recalls))
    print('Average precision on training subset: ', np.mean(train_precisions))
    print('Average f_1 score on training subset: ', np.mean(train_f_1_scores))

    print('Average accuracy on test subset: ', np.mean(test_accuracies))
    print('Average recall on test subset: ', np.mean(test_recalls))
    print('Average precision on test subset: ', np.mean(test_precisions))
    print('Average f_1 score on test subset: ', np.mean(test_f_1_scores))


if __name__ == '__main__':
    main_script()
