import os
from matplotlib import pyplot as plt
from common.particle_swarm_optim_algorithm import PSO
import numpy
import torch

from income_classifiers.pso import dataset_setup


def main_script():
    random_state = 9
    experiment_id = '9'
    torch.manual_seed(random_state)
    numpy.random.seed(random_state)

    dataset = dataset_setup.setup(random_state)

    population = 50
    search_space_dimensions = (119, 5, 1)
    learning_rate = 0.01
    alpha = 0.8
    epochs = 1000
    num_init_connections = (15, 5)
    epsilon = 0.5
    connections_limits = ((2, 30), (2, 10))
    patience = 25
    early_stopping_threshold = 0.000001
    results_path = os.path.abspath('../python/income_classifiers/pso/results/')

    ''' Structure Learning '''
    pso = PSO(number_of_particles=population, search_space_dimensions=search_space_dimensions, dataset=dataset,
              learning_rate=learning_rate, alpha=alpha, epochs=epochs, num_init_connections=num_init_connections,
              epsilon=epsilon, log=True, connections_limits=connections_limits, results_path=results_path,
              experiment_id=experiment_id, random_state=random_state, patience=patience,
              early_stopping_threshold=early_stopping_threshold)
    pso.initialize()

    print('************** FIRST PHASE **************')
    pso.w = 1
    pso.c_1 = 10
    pso.c_2 = 10
    pso.epochs = 500
    pso.optimize(False, 25)
    plt.scatter(range(population), pso.best_scores, c='b', s=50)

    print('************** SECOND PHASE **************')
    pso.w = 0.9
    pso.c_1 = 10
    pso.c_2 = 10
    pso.epochs = 750
    pso.optimize(False, 25)
    plt.scatter(range(population), pso.best_scores, c='r', s=30)

    print('************** THIRD PHASE **************')
    pso.w = 0.9
    pso.c_1 = 1
    pso.c_2 = 20
    pso.epochs = 1000
    pso.optimize(True, 10)
    plt.scatter(range(population), pso.best_scores, c='c', s=10)

    plt.grid(True)
    plt.xlabel("Particle Number")
    plt.ylabel("Objective Function Value")
    plt.legend(['First Phase', 'Second Phase', 'Third Phase'])
    plt.show()


if __name__ == '__main__':
    main_script()
