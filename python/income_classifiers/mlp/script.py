import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from income_classifiers.pso import dataset_setup


class IncomeMLP(nn.Module):
    def __init__(self, num_neurons, learning_rate, epochs=100):
        super(IncomeMLP, self).__init__()
        self.num_neurons = num_neurons
        self.linear_1 = nn.Linear(119, num_neurons)
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
            # print(loss)
            # loss_dif = loss_dif - loss / loss_dif
            loss.backward()
            self.optimizer.step()

    def evaluate_model(self, train_x, train_y, test_x, test_y, train_accuracy, accuracy, precision, recall, f_1):
        train_predict_out = self.forward(train_x)
        train_predict_y = torch.round(train_predict_out)
        predict_out = self.forward(test_x)
        predict_y = torch.round(predict_out)
        train_accuracy.append(accuracy_score(train_y.data, train_predict_y.data))
        accuracy.append(accuracy_score(test_y.data, predict_y.data))
        precision.append(precision_score(test_y.data, predict_y.data, average='macro'))
        recall.append(recall_score(test_y.data, predict_y.data, average='macro'))
        f_1.append(f1_score(test_y.data, predict_y.data, average='macro'))


def main_script():
    train_accuracy = []
    accuracy = []
    precision = []
    recall = []
    f_1 = []
    for i in range(10):
        random_state = i
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        model = IncomeMLP(5, learning_rate=0.01, epochs=1000)
        dataset = dataset_setup.setup(random_state)

        model.train_model(dataset.train_X, dataset.train_y)
        model.evaluate_model(dataset.train_X, dataset.train_y, dataset.test_X, dataset.test_y, train_accuracy, accuracy,
                             precision, recall, f_1)

    print('Average train accuracy: ', np.mean(train_accuracy))
    print('Average accuracy: ', np.mean(accuracy))
    print('Average recall: ', np.mean(recall))
    print('Average precision: ', np.mean(precision))
    print('Average f_1 score: ', np.mean(f_1))

