import numpy as np
from common.sparsenetworkmodel import EdgeWeightedQBAF
import common.helper as helper
import copy
import time


class PSO:
    """
    This class represents the PSO algorithm with its parameters and functions
    Some of the attributes:
        experiment_id: identifier for the experiment
        velocities: set of particles' velocity matrices
        best_scores: set of particles' personal best scores
        particles: set of particles' models
        positions: set of particles' position matrices
        best_positions: set of particles' personal best positions
        global_best_position: global best position
        w: inertia weight (influence of old velocity)
        c_1: acceleration coefficient (cognitive parameter)
        c_2: acceleration coefficient (social parameter)
        num_init_connections: numbers of initial connections (tuple)
        epsilon: velocity bias parameter
        log: if set True, detailed progress will be printed
        best_model: global best model

    Important notice: in the PSO algorithm we use (m x n) binary connection matrices rather than connectivity matrices
    of the sparselinear library. This makes it easier to work with position and velocity matrices. We use helper
    function for conversions.
    """

    def __init__(self, number_of_particles, search_space_dimensions, dataset, learning_rate, alpha, epochs,
                 num_init_connections, connections_limits, epsilon, log, results_path, experiment_id='1',
                 random_state=42, patience=100, early_stopping_threshold=0.000001):
        self.experiment_id = experiment_id
        self.random_state = random_state
        self.number_of_particles = number_of_particles
        self.search_space_dimensions = search_space_dimensions
        self.velocities = []
        self.best_scores = np.zeros(self.number_of_particles)
        self.particles = []
        self.positions = []
        self.best_positions = []
        self.dataset = dataset
        self.global_best_position = (np.zeros((self.search_space_dimensions[0], self.search_space_dimensions[1])),
                                     np.zeros((self.search_space_dimensions[1], self.search_space_dimensions[2])))
        self.global_best_score = 0
        self.global_best_accuracy = 0
        self.global_best_recall = 0
        self.global_best_precision = 0
        self.global_best_f1score = 0
        self.global_best_connection_num = 0
        self.w = 0.9
        self.c_1 = 2
        self.c_2 = 2
        self.num_init_connections = num_init_connections
        self.connections_limits = connections_limits
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.patience = patience
        self.early_stopping_threshold = early_stopping_threshold
        self.log = log
        self.config_log = []
        self.best_model = None
        self.results_path = results_path
        self.start_time = time.time()

    def init_particles(self):
        """
        initializes particles (sparse MLPs) with position and velocity matrices
        """
        for i in range(self.number_of_particles):
            # partially random position initialization
            connections_1, connections_2 = helper.init_random_connections(self.search_space_dimensions[0],
                                                                          self.search_space_dimensions[1],
                                                                          self.search_space_dimensions[2],
                                                                          self.num_init_connections)
            model = EdgeWeightedQBAF(connections_1=connections_1, connections_2=connections_2,
                                     num_features=self.search_space_dimensions[0],
                                     num_neurons=self.search_space_dimensions[1],
                                     num_targets=self.search_space_dimensions[2],
                                     model_number=i, learning_rate=self.learning_rate, alpha=self.alpha,
                                     epochs=self.epochs, dataset=self.dataset, patience=self.patience,
                                     early_stopping_threshold=self.early_stopping_threshold)
            binary_matrix_1 = np.zeros((self.search_space_dimensions[0], self.search_space_dimensions[1]))
            binary_matrix_1 = helper.build_binary_matrix_out_of_connectivity_matrix(binary_matrix_1, connections_1)
            binary_matrix_2 = np.zeros((self.search_space_dimensions[1], self.search_space_dimensions[2]))
            binary_matrix_2 = helper.build_binary_matrix_out_of_connectivity_matrix(binary_matrix_2, connections_2)
            self.positions.append((binary_matrix_1, binary_matrix_2))
            self.best_positions.append((binary_matrix_1, binary_matrix_2))
            self.particles.append(model)
            self.velocities.append((np.zeros((self.search_space_dimensions[0], self.search_space_dimensions[1])),
                                    np.zeros((self.search_space_dimensions[1], self.search_space_dimensions[2]))))

    def evaluate_particle(self, particle):
        """
        Evaluate particle by calculating its score based on the objective function.
        """
        particle.train_and_evaluate_model()
        if self.log:
            print('Particle: ', particle.model_number, ' ||    score: ', particle.score, ',    accuracy: ',
                  particle.test_accuracy, ',    connections: ', particle.num_connections_tuple)

    def update_score(self, particle):
        """
        Updates the personal best score if it is greater than the current one
        Personal best position is also updated in this case
        """
        if particle.score > self.best_scores[particle.model_number]:
            self.best_scores[particle.model_number] = particle.score
            binary_matrix_1 = helper.build_binary_matrix_out_of_connectivity_matrix(
                np.zeros((self.search_space_dimensions[0], self.search_space_dimensions[1])), particle.connections_1)
            binary_matrix_2 = helper.build_binary_matrix_out_of_connectivity_matrix(
                np.zeros((self.search_space_dimensions[1], self.search_space_dimensions[2])), particle.connections_2)
            self.best_positions[particle.model_number] = (binary_matrix_1, binary_matrix_2)

    def update_global_best(self):
        """
        Updates the global best score if a greater one is found
        Other global best attributes are updated accordingly
        """
        maximum = max(self.best_scores)
        if maximum > self.global_best_score:
            idx = int(np.argmax(self.best_scores))
            self.global_best_position = copy.deepcopy(self.positions[idx])
            self.global_best_score = maximum
            self.global_best_accuracy = self.particles[idx].test_accuracy
            self.global_best_connection_num = self.particles[idx].num_connections_tuple

    def update_velocity(self, particle):
        """
        Calculates and updates velocity matrices of the two layers
        """
        self.update_vel_layer(particle, 0)
        self.update_vel_layer(particle, 1)

    def update_vel_layer(self, particle, layer_num):
        """
        Calculates and updates velocity values in the velocity matrix that relates to a specific layer
        See literature for more information
        """
        for i in range(self.search_space_dimensions[0 + layer_num]):
            for j in range(self.search_space_dimensions[1 + layer_num]):
                prev_vel = self.velocities[particle.model_number][layer_num][i][j]  # previous velocity value
                local_best_term = self.best_positions[particle.model_number][layer_num][i][j] - \
                                  self.positions[particle.model_number][layer_num][i][j]
                global_best_term = self.global_best_position[layer_num][i][j] - \
                                   self.positions[particle.model_number][layer_num][i][j]
                r_1 = np.random.uniform(0, 1)
                r_2 = np.random.uniform(0, 1)
                # velocity update equation
                self.velocities[particle.model_number][layer_num][i][j] \
                    = self.w * prev_vel + self.c_1 * r_1 * local_best_term + self.c_2 * r_2 * global_best_term

    def update_position(self, particle):
        """
        Updates the position matrices of a particle
        Afterwards, a new model is constructed based on the new position (connections)
        """
        updated_connections_1 = self.update_layer(particle, 0, self.connections_limits[0])
        updated_connections_2 = self.update_layer(particle, 1, self.connections_limits[1])
        updated_model = EdgeWeightedQBAF(connections_1=updated_connections_1, connections_2=updated_connections_2,
                                         num_features=self.search_space_dimensions[0],
                                         num_neurons=self.search_space_dimensions[1],
                                         num_targets=self.search_space_dimensions[2],
                                         model_number=particle.model_number, learning_rate=self.learning_rate,
                                         alpha=self.alpha, epochs=self.epochs, dataset=self.dataset,
                                         patience=self.patience,
                                         early_stopping_threshold=self.early_stopping_threshold)
        self.particles[particle.model_number] = updated_model

    def update_layer(self, particle, layer_num, connections_limit):
        """
        Updates a connections layer (position).
        See literature for more details
        """
        curr_conn_num = particle.num_connections_tuple[layer_num]  # current connection number
        for i in range(self.search_space_dimensions[0 + layer_num]):
            for j in range(self.search_space_dimensions[1 + layer_num]):
                # position bit update equations with the usage of velocity bias (epsilon)
                probability = 1 / (1 + np.exp(- self.velocities[particle.model_number][layer_num][i][j])) \
                              - self.epsilon
                r_id = np.random.uniform(0, 1)
                if r_id < probability:
                    if curr_conn_num < connections_limit[1]:  # check upper boundary conformity
                        if self.positions[particle.model_number][layer_num][i][j] == 0:
                            curr_conn_num = curr_conn_num + 1
                            self.positions[particle.model_number][layer_num][i][j] = 1
                else:
                    if curr_conn_num > connections_limit[0]:  # ensure at least one connection exists (lower boundary)
                        if self.positions[particle.model_number][layer_num][i][j] == 1:
                            curr_conn_num = curr_conn_num - 1
                            self.positions[particle.model_number][layer_num][i][j] = 0
        return helper.build_sparse_connectivity_out_of_binary_matrix(
            self.positions[particle.model_number][layer_num])

    def initialize(self):
        self.init_particles()

    def optimize(self, final, iterations):
        """
        This function represents the optimization sub-routine. It runs the optimization for a specific number of
        iterations
        """
        self.log_config(iterations)
        for i in range(iterations):
            print("Iteration: ", i)
            for p in self.particles:
                self.evaluate_particle(p)
                self.update_score(p)
            self.update_global_best()
            print('Best particle: ', self.global_best_score, " || accuracy: ", self.global_best_accuracy,
                  ', connections: ', self.global_best_connection_num)
            if i == iterations - 1 and final:
                self.finalize()
                break
            for p in self.particles:
                self.update_velocity(p)
                self.update_position(p)

    def log_config(self, iterations):
        self.config_log.append('iterations: ' + str(iterations) + ', w=' + str(self.w) +
                               ', c_1=' + str(self.c_1) + ', c_2=' + str(self.c_2) +
                               ', epsilon=' + str(self.epsilon))

    def finalize(self):
        """
        Executed as a final step. The global best model is constructed, trained and evaluated.
        A report of the experiment is persisted as a txt file
        Additionally, binary connection matrices are persisted, in order to be able to construct the model again outside
        the PSO algorithm
        """
        conn_1 = helper.build_sparse_connectivity_out_of_binary_matrix(self.global_best_position[0])
        conn_2 = helper.build_sparse_connectivity_out_of_binary_matrix(self.global_best_position[1])
        self.best_model = EdgeWeightedQBAF(connections_1=conn_1, connections_2=conn_2,
                                           num_features=self.search_space_dimensions[0],
                                           num_neurons=self.search_space_dimensions[1],
                                           num_targets=self.search_space_dimensions[2],
                                           model_number=0, learning_rate=self.learning_rate,
                                           alpha=self.alpha, epochs=self.epochs, dataset=self.dataset,
                                           patience=self.patience,
                                           early_stopping_threshold=self.early_stopping_threshold)
        self.best_model.train_and_evaluate_model()
        self.best_model.evaluate_model_final()

        connected_features = helper.get_connected_features(self.global_best_position[0])
        connected_features_names = [self.dataset.feature_names[i] for i in connected_features]

        report_file = open(self.results_path + '/' + self.experiment_id + ".txt", "w")
        report_file.write("experiment_num: " + self.experiment_id)
        report_file.write("\n")
        report_file.write("\nParameters:")
        report_file.write("\nrandom_state=" + str(self.random_state))
        report_file.write("\npopulation=" + str(self.number_of_particles))
        report_file.write("\nsearch_space_dimensions=" + str(self.search_space_dimensions))
        report_file.write("\nlearning_rate=" + str(self.learning_rate))
        report_file.write("\nalpha=" + str(self.alpha))
        report_file.write("\nepochs=" + str(self.epochs))
        report_file.write("\npatience=" + str(self.patience))
        report_file.write("\nearly_stopping_threshold=" + str(self.early_stopping_threshold))
        report_file.write("\nnum_init_connections=" + str(self.num_init_connections))
        report_file.write("\nconnections_limits=" + str(self.connections_limits))
        report_file.write("\n")
        report_file.write("\n")

        report_file.write("\nConfigurations:")
        for i in self.config_log:
            report_file.write("\n" + i)
        report_file.write("\n")
        report_file.write("\n")

        report_file.write("\nbest particle:")
        report_file.write("\ntrain accuracy: " + str(self.best_model.train_accuracy))
        report_file.write("\ntest accuracy: " + str(self.best_model.test_accuracy))
        report_file.write("\ntest recall: " + str(self.best_model.recall))
        report_file.write("\ntest precision: " + str(self.best_model.precision))
        report_file.write("\ntest f1 score: " + str(self.best_model.f1_score))
        report_file.write("\n\nnumber of connections: " + str(self.best_model.num_connections_tuple))
        report_file.write("\n\nconnected features:")
        for i in connected_features_names:
            report_file.write("\n")
            report_file.write(str(i))

        report_file.write("\n\nfirst connectivity matrix:\n")
        report_file.write(np.array2string(self.global_best_position[0].astype(int)))
        report_file.write("\n")
        report_file.write("\nsecond connectivity matrix:\n")
        report_file.write(np.array2string(self.global_best_position[1].astype(int)))
        report_file.write("\n")
        report_file.write("\nfirst layer weights:\n")
        report_file.write(str(self.best_model.sparse_linear_1.weight))
        report_file.write("\nfirst layer biases:\n")
        report_file.write(str(self.best_model.sparse_linear_1.bias))
        report_file.write("\nsecond layer weights:\n")
        report_file.write(str(self.best_model.sparse_linear2.weight))
        report_file.write("\nsecond layer biases:\n")
        report_file.write(str(self.best_model.sparse_linear2.bias))

        report_file.write("\n\nRuntime: %.2f seconds" % (time.time() - self.start_time))

        report_file.close()

        np.savetxt(self.results_path + '/connections_' + self.experiment_id + '_1.txt', self.global_best_position[0])
        np.savetxt(self.results_path + '/connections_' + self.experiment_id + '_2.txt', self.global_best_position[1])
