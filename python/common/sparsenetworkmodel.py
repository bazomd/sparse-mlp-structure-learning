from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import sparselinear


class EdgeWeightedQBAF(nn.Module):
    """
    This class represents a sparse MLP model that can be represented as a QBAF. It also represents a PSO particle.
    Some attributes:
        model_number: used as an identifier for the model (particle)
        num_neurons: number of hidden nodes
        num_targets: number of target features
        connections_1, connections_2: connectivity matrices for the sparselinear layers
        num_connections: number of overall existing connections
        num_connections_tuple: number of existing connections in each layer
        num_total_connections: number of overall possible connections
        num_total_conn_tuple: number of possible connections in each layer
    """
    def __init__(self, dataset, connections_1, connections_2, num_features, num_neurons, num_targets, model_number,
                 learning_rate, alpha, patience=25, epochs=50, train_samples_ratio=1.0,
                 early_stopping_threshold=0.000001):
        super(EdgeWeightedQBAF, self).__init__()
        self.model_number = model_number

        # initialize the network layers and functions
        self.num_neurons = num_neurons
        self.sparse_linear_1 = sparselinear.SparseLinear(num_features, num_neurons, connectivity=connections_1)
        self.sigmoid = nn.Sigmoid()
        self.sparse_linear2 = sparselinear.SparseLinear(num_neurons, num_targets, connectivity=connections_2)

        # structure attributes
        self.num_targets = num_targets
        self.connections_1 = connections_1
        self.connections_2 = connections_2
        self.num_connections = self.connections_1.shape[1] + self.connections_2.shape[1]
        self.num_connections_tuple = (self.connections_1.shape[1], self.connections_2.shape[1])
        self.num_total_conn_tuple = (num_features * self.num_neurons, num_targets * self.num_neurons)
        self.num_total_connections = num_features * self.num_neurons + num_targets * self.num_neurons

        # learning and performance attributes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = None
        self.criterion = None
        self.patience = patience
        self.early_stopping_threshold = early_stopping_threshold
        self.alpha = alpha
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.score = 0
        self.dataset = dataset
        self.train_samples_ratio = train_samples_ratio

    def forward(self, x):
        """
        Feed-forward x through the network
        """
        x = self.sparse_linear_1.forward(x)
        x = self.sigmoid.forward(x)
        x = self.sparse_linear2.forward(x)
        if self.num_targets == 1:
            '''for binary classification: softmax is the same as sigmoid. softmax on multi-class are implicitly used in
            the BCELoss'''
            x = self.sigmoid.forward(x)
        return x

    def train_and_evaluate_model(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.num_targets == 1:
            # different loss functions, depending on number of targets
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        best_score = None
        patience_counter = 0
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            train_predictions = self.forward(self.dataset.train_X)
            if self.num_targets == 1:
                train_predictions = train_predictions.reshape(1, -1)[0]
            loss = self.criterion(train_predictions, self.dataset.train_y)
            loss.backward()
            self.optimizer.step()
            # check if early stopping required
            train_loss = loss.item()
            if best_score is None:
                best_score = train_loss
            elif best_score - train_loss < self.early_stopping_threshold:
                patience_counter += 1
            else:
                best_score = train_loss
                patience_counter = 0
            if patience_counter >= self.patience:
                break
        self.evaluate_model()

    def evaluate_model(self):
        """
        Procedure for evaluating the model after the training process
        """
        predict_out = self.forward(self.dataset.test_X)
        if self.num_targets == 1:
            predict_y = torch.round(predict_out)
        else:
            _, predict_y = torch.max(predict_out, 1)
        self.test_accuracy = accuracy_score(self.dataset.test_y.data, predict_y.data)
        # calculate the score based on the PSO objective function
        self.score = self.alpha * self.test_accuracy + (1 - self.alpha) * (
                (self.num_total_connections - self.num_connections) / self.num_total_connections)

    def evaluate_model_final(self):
        """
        Process for evaluating the model after the PSO finishes. Here, additional performance scores are relevant, such
        as recall and precision. Also, the train accuracy is calculated only here, since it is not relevant for the PSO.
        """
        train_predict_out = self.forward(self.dataset.train_X)
        test_predict_out = self.forward(self.dataset.test_X)
        if self.num_targets == 1:
            train_predict_y = torch.round(train_predict_out)
            test_predict_y = torch.round(test_predict_out)
        else:
            _, train_predict_y = torch.max(train_predict_out, 1)
            _, test_predict_y = torch.max(test_predict_out, 1)
        # calculate all performance scores
        self.train_accuracy = accuracy_score(self.dataset.train_y.data, train_predict_y.data)
        self.test_accuracy = accuracy_score(self.dataset.test_y.data, test_predict_y.data)
        self.precision = precision_score(self.dataset.test_y.data, test_predict_y.data, average='macro')
        self.recall = recall_score(self.dataset.test_y.data, test_predict_y.data, average='macro')
        self.f1_score = f1_score(self.dataset.test_y.data, test_predict_y.data, average='macro')
        # calculate the score based on the PSO objective function
        self.score = self.alpha * self.test_accuracy + (1 - self.alpha) * (
                (self.num_total_connections - self.num_connections) / self.num_total_connections)
