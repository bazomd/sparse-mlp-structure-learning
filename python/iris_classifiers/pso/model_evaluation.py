import os

import numpy
import torch

from common import helper
from common.sparsenetworkmodel import EdgeWeightedQBAF
from iris_classifiers.pso import dataset_setup


def main_script():
    random_state = 42
    torch.manual_seed(random_state)
    numpy.random.seed(random_state)

    BIN_SIZE = 10
    dataset = dataset_setup.setup(random_state=random_state, bin_size=BIN_SIZE, discretization_strategy='quantile')
    results_path = os.path.abspath('../python/iris_classifiers/pso/results/')

    connections_1 = numpy.loadtxt(results_path + '/connections_0_1.txt')
    connections_2 = numpy.loadtxt(results_path + '/connections_0_2.txt')
    conn_1 = helper.build_sparse_connectivity_out_of_binary_matrix(connections_1)
    conn_2 = helper.build_sparse_connectivity_out_of_binary_matrix(connections_2)

    nn = EdgeWeightedQBAF(dataset=dataset, connections_1=conn_1, connections_2=conn_2, num_features=12, num_neurons=6,
                          num_targets=3, model_number=0, learning_rate=0.01, alpha=0.7, epochs=500, patience=10,
                          early_stopping_threshold=0.0001)

    nn.train_and_evaluate_model()
    nn.evaluate_model_final()
    print('Connections: ', nn.num_connections_tuple)
    print('Train accuracy: ', nn.test_accuracy)
    print('Test accuracy: ', nn.test_accuracy)
    print('recall: ', nn.recall)
    print('precision: ', nn.precision)
    print('f1_score: ', nn.f1_score)

    print(nn.sparse_linear_1.weight)
    print("\nfirst layer biases:\n")
    print(nn.sparse_linear_1.bias)
    print("\nsecond layer weights:\n")
    print(nn.sparse_linear2.weight)
    print("\nsecond layer biases:\n")
    print(nn.sparse_linear2.bias)


if __name__ == '__main__':
    main_script()
