import torch
import numpy as np

"""
This script contains helper functions. 
"""


def init_random_connections(num_in, num_neurons, num_out, num_init_connections):
    """
    Functions that initializes connections (position) in a particle
    :param num_in: number of input nodes
    :param num_neurons: number of hidden nodes
    :param num_out: number of output nodes
    :param num_init_connections: tuple of numbers of initial connections in each layer
    :return: two connectivity matrices with randomly distributed connections in each one
    """
    number_of_connections = num_init_connections[1]
    bin_matrix_2 = np.zeros((num_neurons, num_out))  # init binary matrix for the second layer
    while number_of_connections != 0:
        i = np.random.randint(0, num_neurons)
        j = np.random.randint(0, num_out)
        if bin_matrix_2[i][j] != 1:  # avoid reoccurrence of bit
            bin_matrix_2[i][j] = 1
            number_of_connections = number_of_connections - 1
    # fill second layer
    number_of_connections = num_init_connections[0]
    bin_matrix_1 = np.zeros((num_in, num_neurons))  # init binary matrix for the first layer
    # next: ensure that only hidden nodes that are connected in the second layer are connected in the first layer
    connected_neurons = []
    for k in range(num_neurons):
        if 1 in bin_matrix_2[k][:]:
            connected_neurons.append(k)
    while number_of_connections != 0:
        i = np.random.randint(0, num_in)
        j = np.random.choice(connected_neurons, 1)[0]
        if bin_matrix_1[i][j] != 1:
            bin_matrix_1[i][j] = 1
            number_of_connections = number_of_connections - 1
    mat_1 = build_sparse_connectivity_out_of_binary_matrix(bin_matrix_1)
    mat_2 = build_sparse_connectivity_out_of_binary_matrix(bin_matrix_2)
    return mat_1, mat_2


def build_sparse_connectivity_out_of_binary_matrix(binary_matrix):
    """
    Used for converting a binary connection matrix (m x n) into a connectivity matrix (2 x k) that can be used in the
    sparselinear library
    :param binary_matrix: (m x n) binary matrix, with m: number of nodes of first layer and n: number of nodes in second
        layer
    :return: connectivity matrix (2 x k), with  k: number of connections. See docs of sparselinear for more info
    """
    sources = np.array([])
    destinations = np.array([])
    for i in range(binary_matrix.shape[0]):
        for j in range(binary_matrix.shape[1]):
            if binary_matrix[i][j] == 1:
                sources = np.append(sources, i)
                destinations = np.append(destinations, j)
    sources = torch.from_numpy(sources.reshape(1, -1)).long()
    destinations = torch.from_numpy(destinations.reshape(1, -1)).long()
    model_connectivity = torch.cat((destinations, sources), dim=0)
    return model_connectivity


def build_sparse_connectivity_out_of_binary_vector(binary_vector):
    """
    Used for constructing a connectivity matrix out of binary connection vector. This useful when we have only one node
        as a target feature. The binary connection matrix will be then a vector
    :param binary_vector: represents the connected nodes in the previous layer
    :return: connectivity matrix
    """
    sources = np.array([])
    destinations = np.array([])
    for i in range(binary_vector.shape[0]):
        if binary_vector[i]:
            sources = np.append(sources, i)
            destinations = np.append(destinations, 0)  # only zero since we have one node to connect to
    sources = torch.from_numpy(sources.reshape(1, -1)).long()
    destinations = torch.from_numpy(destinations.reshape(1, -1)).long()
    model_connectivity = torch.cat((destinations, sources), dim=0)
    return model_connectivity


def build_binary_matrix_out_of_connectivity_matrix(binary_matrix, model_connectivity):
    """
    convert connectivity matrix into a binary connection matrix. See doc of the function:
        build_sparse_connectivity_out_of_binary_matrix
    :param binary_matrix: binary connection matrix
    :param model_connectivity: connectivity matrix
    :return: binary connection matrix, filled according to model_connectivity
    """
    for i in range(model_connectivity.shape[1]):
        binary_matrix[model_connectivity[1][i]][model_connectivity[0][i]] = 1
    return binary_matrix


def get_connected_features(connection_matrix):
    """
    Extract indices of connected features out of binary connection matrix
    :param connection_matrix: binary connection matrix
    :return: array of connected features indices
    """
    connected_features = []
    for i in range(connection_matrix.shape[0]):
        if np.count_nonzero(connection_matrix[i, :]) > 0:  # check if any nodes are connected
            connected_features.append(i)
    return connected_features
