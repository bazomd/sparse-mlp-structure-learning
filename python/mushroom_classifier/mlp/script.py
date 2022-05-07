import numpy
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mushroom_classifier.pso import dataset_setup

"""
In this script, a fully connected MLP is constructed, trained, and evaluated.
"""


class MushroomMLP(nn.Module):
    """
    MLP model
    """
    def __init__(self, num_neurons, learning_rate, epochs=50):
        super(MushroomMLP, self).__init__()
        self.num_neurons = num_neurons
        self.linear_1 = nn.Linear(111, num_neurons)
        self.sigmoid = nn.Sigmoid()
        self.linear_2 = nn.Linear(num_neurons, 1)

        self.learning_rate = learning_rate
        self.optimizer = None
        self.criterion = None
        self.epochs = epochs

    def forward(self, x):
        x = self.linear_1.forward(x)
        x = self.sigmoid.forward(x)
        x = self.linear_2.forward(x)
        x = self.sigmoid.forward(x)
        return x

    def train_model(self, train_x, train_y):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()
        self.optimizer.zero_grad()
        train_predictions = self.forward(train_x)
        train_predictions = train_predictions.reshape(1, -1)[0]
        loss = self.criterion(train_predictions, train_y)
        # loss_dif = loss
        loss.backward()
        self.optimizer.step()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            train_predictions = self.forward(train_x)
            train_predictions = train_predictions.reshape(1, -1)[0]
            loss = self.criterion(train_predictions, train_y)
            # print(loss_dif)
            # loss_dif = loss_dif - loss / loss_dif
            loss.backward()
            self.optimizer.step()

    def evaluate_model(self, train_x, train_y, test_x, test_y, train_accuracy, accuracy, precision, recall, f_1):
        train_predict_out = self.forward(train_x)
        train_predict_y = torch.round(train_predict_out)
        train_accuracy.append(accuracy_score(train_y.data, train_predict_y.data))
        predict_out = self.forward(test_x)
        predict_y = torch.round(predict_out)
        accuracy.append(accuracy_score(test_y.data, predict_y.data))
        precision.append(precision_score(test_y.data, predict_y.data))
        recall.append(recall_score(test_y.data, predict_y.data))
        f_1.append(f1_score(test_y.data, predict_y.data, average='micro'))


def main_script():
    train_accuracy = []
    accuracy = []
    precision = []
    recall = []
    f_1 = []
    # construct and evaluate ten models with ten different random seeds and calculate the averaged results
    for i in range(10):
        random_state = i
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        model = MushroomMLP(5, learning_rate=0.01, epochs=300)
        dataset = dataset_setup.setup(random_state)

        model.train_model(dataset.train_X, dataset.train_y)
        model.evaluate_model(dataset.train_X, dataset.train_y, dataset.test_X, dataset.test_y, train_accuracy, accuracy,
                             precision, recall, f_1)

    print('Average train accuracy: ', numpy.mean(train_accuracy))
    print('Average accuracy: ', numpy.mean(accuracy))
    print('Average recall: ', numpy.mean(recall))
    print('Average precision: ', numpy.mean(precision))
    print('Average f_1 score: ', numpy.mean(f_1))


if __name__ == '__main__':
    main_script()
