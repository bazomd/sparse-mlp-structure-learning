import os

import numpy
import torch
from matplotlib import pyplot as plt

from common.particle_swarm_optim_algorithm import PSO
from mushroom_classifier.pso import dataset_setup


def main_script():
    random_state = 0
    experiment_id = '0'  # identifier

    # fix random seed for all libraries
    torch.manual_seed(random_state)
    numpy.random.seed(random_state)

    # set up the dataset
    dataset = dataset_setup.setup(random_state)

    # hyperparameter values
    population = 100
    search_space_dimensions = (111, 5, 1)
    learning_rate = 0.01
    alpha = 0.7
    epochs = 100
    num_init_connections = (15, 5)
    epsilon = 0.5
    connections_limits = ((2, 15), (2, 5))
    patience = 10
    early_stopping_threshold = 0.001

    # output path for result files
    results_path = os.path.abspath('../python/mushroom_classifier/pso/results/')

    # construct the PSO algorithm
    pso = PSO(number_of_particles=population, search_space_dimensions=search_space_dimensions, dataset=dataset,
              learning_rate=learning_rate, alpha=alpha, epochs=epochs, num_init_connections=num_init_connections,
              epsilon=epsilon, log=True,
              connections_limits=connections_limits, results_path=results_path, experiment_id=experiment_id,
              random_state=random_state, patience=patience, early_stopping_threshold=early_stopping_threshold)

    pso.initialize()

    print('************** FIRST PHASE **************')
    pso.w = 0.9
    pso.c_1 = 2
    pso.c_2 = 2
    pso.epsilon = 0.5
    pso.optimize(False, 20)
    # plot progress of best personal scores after each optimization phase
    plt.scatter(range(population), pso.best_scores, c='b', s=50)

    print('************** SECOND PHASE **************')
    pso.w = 0.9
    pso.c_1 = 4
    pso.c_2 = 4
    pso.epochs = 200
    pso.epsilon = 0.5
    pso.optimize(False, 50)
    plt.scatter(range(population), pso.best_scores, c='r', s=30)

    print('************** THIRD PHASE **************')
    pso.w = 0.9
    pso.c_1 = 0.5
    pso.c_2 = 10
    pso.epochs = 300
    pso.epsilon = 0.5
    pso.optimize(True, 30)
    plt.scatter(range(population), pso.best_scores, c='c', s=10)

    # persist the plot
    plt.grid(True)
    plt.xlabel("Particle Number")
    plt.ylabel("Objective Function Value")
    plt.legend(['First Phase', 'Second Phase', 'Third Phase'])
    plt.savefig(results_path + '/plot_' + experiment_id + '.png')
    plt.show()


if __name__ == '__main__':
    main_script()
