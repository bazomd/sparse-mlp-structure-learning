import os

from matplotlib import pyplot as plt
from common.particle_swarm_optim_algorithm import PSO
from iris_classifiers.pso import dataset_setup
import numpy
import torch


def main_script():
    random_state = 42
    torch.manual_seed(random_state)
    numpy.random.seed(random_state)

    BIN_SIZE = 10
    DISCRETIZATION_STRATEGY = 'quantile'  # {'uniform', 'quantile', 'kmeans'}
    dataset = dataset_setup.setup(random_state, bin_size=BIN_SIZE, discretization_strategy=DISCRETIZATION_STRATEGY)

    experiment_id = '42'
    population = 30
    search_space_dimensions = (12, 6, 3)
    learning_rate = 0.01
    alpha = 0.7
    epochs = 450
    num_init_connections = (7, 3)
    epsilon = 0.4
    connections_limits = ((2, 10000), (2, 100000))
    patience = 10
    early_stopping_threshold = 0.0001
    results_path = os.path.abspath('../python/iris_classifiers/pso/results/')

    pso = PSO(number_of_particles=population, search_space_dimensions=search_space_dimensions, dataset=dataset,
              learning_rate=learning_rate, alpha=alpha, epochs=epochs, num_init_connections=num_init_connections,
              epsilon=epsilon, log=True, connections_limits=connections_limits,
              results_path=results_path, experiment_id=experiment_id, random_state=random_state, patience=patience,
              early_stopping_threshold=early_stopping_threshold)

    pso.initialize()

    print('************** FIRST PHASE **************')
    pso.w = 0.9
    pso.c_1 = 2
    pso.c_2 = 2
    pso.epsilon = 0.4
    pso.optimize(False, 30)
    plt.scatter(range(population), pso.best_scores, c='b', s=50)

    print('************** SECOND PHASE **************')
    pso.w = 0.9
    pso.c_1 = 4
    pso.c_2 = 4
    pso.epsilon = 0.4
    pso.optimize(False, 30)
    plt.scatter(range(population), pso.best_scores, c='r', s=30)

    print('************** THIRD PHASE **************')
    pso.w = 0.9
    pso.c_1 = 1
    pso.c_2 = 10
    pso.epsilon = 0.4
    pso.optimize(True, 20)
    plt.scatter(range(population), pso.best_scores, c='c', s=10)

    plt.grid(True)
    plt.xlabel("Particle Number")
    plt.ylabel("Objective Function Value")
    plt.legend(['First Phase', 'Second Phase', 'Third Phase'])
    plt.savefig(results_path + '/plot_' + experiment_id + '.png')
    plt.show()


if __name__ == '__main__':
    main_script()
