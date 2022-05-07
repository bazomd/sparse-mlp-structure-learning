import os

import numpy
import torch

from common import helper
from common.sparsenetworkmodel import EdgeWeightedQBAF
from mushroom_classifier.pso import dataset_setup


def main_script():
    """
    Construct a sparse MLP (QBAF) out of the binary connection matrices that were persisted as a result of PSO
    The model is then trained and evaluated
    """

    random_state = 6
    torch.manual_seed(random_state)
    numpy.random.seed(random_state)

    dataset = dataset_setup.setup(random_state)

    results_path = os.path.abspath('../python/mushroom_classifier/pso/results/')

    connections_1 = numpy.loadtxt(results_path + '/connections_6_1.txt')
    connections_2 = numpy.loadtxt(results_path + '/connections_6_2.txt')
    conn_1 = helper.build_sparse_connectivity_out_of_binary_matrix(connections_1)
    conn_2 = helper.build_sparse_connectivity_out_of_binary_vector(connections_2)

    nn = EdgeWeightedQBAF(connections_1=conn_1, connections_2=conn_2, num_features=111, num_neurons=5, num_targets=1,
                          model_number=0, learning_rate=0.01, alpha=0.7, epochs=450, dataset=dataset, patience=450)

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
