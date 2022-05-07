import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import datasets
from sklearn import model_selection
from torch.autograd import Variable
import numpy


class IrisMLP(nn.Module):
    def __init__(self, num_neurons, learning_rate, epochs=50):
        super(IrisMLP, self).__init__()
        self.num_neurons = num_neurons
        self.linear_1 = nn.Linear(4, num_neurons)
        self.sigmoid = nn.Sigmoid()
        self.linear_2 = nn.Linear(num_neurons, 3)

        self.learning_rate = learning_rate
        self.optimizer = None
        self.criterion = None
        self.epochs = epochs

    def forward(self, x):
        x = self.linear_1.forward(x)
        x = self.sigmoid.forward(x)
        x = self.linear_2.forward(x)
        return x

    def train_model(self, train_x, train_y):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            train_predictions = self.forward(train_x)
            loss = self.criterion(train_predictions, train_y)
            # print(loss)
            loss.backward()
            self.optimizer.step()

    def evaluate_model(self, train_x, train_y, test_x, test_y, train_accuracy, accuracy, precision, recall, f_1):
        train_predict_out = self.forward(train_x)
        _, train_predict_y = torch.max(train_predict_out, 1)
        train_accuracy.append(accuracy_score(train_y.data, train_predict_y.data))
        predict_out = self.forward(test_x)
        _, predict_y = torch.max(predict_out, 1)
        accuracy.append(accuracy_score(test_y.data, predict_y.data))
        precision.append(precision_score(test_y.data, predict_y.data, average='micro'))
        recall.append(recall_score(test_y.data, predict_y.data, average='micro'))
        f_1.append(f1_score(test_y.data, predict_y.data, average='micro'))


def main_script():
    train_accuracy = []
    accuracy = []
    precision = []
    recall = []
    f_1 = []
    for i in range(10):
        random_state = i
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        model = IrisMLP(2, learning_rate=0.01, epochs=450)
        dataset = datasets.load_iris()
        X = dataset['data']
        y = dataset['target']
        train_X, test_X, train_y, test_y = model_selection.train_test_split(X, y, test_size=0.2)

        train_X = Variable(torch.Tensor(train_X).float(), requires_grad=False)
        test_X = Variable(torch.Tensor(test_X).float(), requires_grad=False)
        train_y = Variable(torch.Tensor(train_y).long(), requires_grad=False)
        test_y = Variable(torch.Tensor(test_y).long(), requires_grad=False)

        model.train_model(train_X, train_y)
        model.evaluate_model(train_X, train_y, test_X, test_y, train_accuracy, accuracy, precision, recall, f_1)

    print('Average train accuracy: ', numpy.mean(train_accuracy))
    print('Average accuracy: ', numpy.mean(accuracy))
    print('Average recall: ', numpy.mean(recall))
    print('Average precision: ', numpy.mean(precision))
    print('Average f_1 score: ', numpy.mean(f_1))



if __name__ == '__main__':
    main_script()
