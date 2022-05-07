import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main_script():
    max_depth = 7

    dataset = datasets.load_iris()

    x = dataset.data
    y = dataset.target

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
        #                       feature_names=dataset.feature_names,
        #                       class_names=dataset.target_names,
        #                       filled=True)
        # fig.savefig("iris_decistion_tree.png")

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
