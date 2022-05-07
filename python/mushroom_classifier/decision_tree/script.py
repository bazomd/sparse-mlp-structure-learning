import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import graphviz
import pydot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main_script():
    """
    Construct and evaluate decision trees with different depths
    """
    max_depth = 3

    # preprocessing
    dataset = pd.read_csv('../python/mushroom_classifier/dataset/agaricus-lepiota.data')

    # label_encoder = LabelEncoder()
    # for column in dataset.columns:
    #     dataset[column] = label_encoder.fit_transform(dataset[column])

    y = dataset['class']
    x = dataset.drop(['class'], axis=1)

    # convert categorical variable into dummy/indicator variables
    x = pd.get_dummies(x)
    y = pd.get_dummies(y)

    test_accuracies = []
    test_recalls = []
    test_precisions = []
    test_f_1_scores = []

    train_accuracies = []
    train_recalls = []
    train_precisions = []
    train_f_1_scores = []
    for i in range(1):
        random_state = 42
        np.random.seed(random_state)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

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

        # fig = plt.figure(figsize=(25, 20))
        # plot_ = tree.plot_tree(model,
        #                       feature_names=X_test.columns.tolist(),
        #                       class_names=y_test.columns.tolist(),
        #                       filled=True)
        # fig.savefig("mushroom_decistion_tree.png")
        # source = tree.export_graphviz(model, out_file=None,
        #                              feature_names=X_train.columns,
        #                              filled=True, rounded=True,
        #                              special_characters=True)
        # graph = graphviz.Source(source)
        # graph.save()

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
