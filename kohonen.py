import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches


class Kohonen:

    def __init__(self, n_iterations, init_learning_rate, training_set) -> None:

        network_dimensions = np.array([7, 7])
        self.n_iterations = n_iterations
        self.init_learning_rate = init_learning_rate

        normalise_data = False
        normalise_by_column = False

        self.m = training_set.shape[0]
        self.n = training_set.shape[1]

        # initial neighbourhood radius
        self.init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
        # radius decay parameter
        self.time_constant = self.n_iterations / np.log(self.init_radius)

        self.data = training_set
        if normalise_data:
            if normalise_by_column:
                col_maxes = training_set.max(axis=0)
                self.data = training_set / col_maxes[np.newaxis, :]
            else:
                self.data = training_set / self.data.max()

        self.net = np.random.random((network_dimensions[0], network_dimensions[1], self.m))

    def train(self):
        for i in range(self.n_iterations):
            # select a training example at random
            t = self.data[:, np.random.randint(0, self.n)].reshape(np.array([self.m, 1]))

            # find its Best Matching Unit
            bmu, bmu_idx = self.find_bmu(t)

            if i == self.n_iterations-1:
                print(bmu_idx)

            # decay the SOM parameters
            r = self.decay_radius(i)
            l = self.decay_learning_rate(i)

            # update weight vector to move closer to input
            # and move its neighbours in 2-D vector space closer

            for x in range(self.net.shape[0]):
                for y in range(self.net.shape[1]):
                    w = self.net[x, y, :].reshape(self.m, 1)
                    w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                    w_dist = np.sqrt(w_dist)

                    if w_dist <= r:
                        # calculate the degree of influence (based on the 2-D distance)
                        influence = Kohonen.calculate_influence(w_dist, r)

                        # new w = old w + (learning rate * influence * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (l * influence * (t - w))
                        self.net[x, y, :] = new_w.reshape(1, len(t))

    def find_bmu(self, t):
        """
            Find the best matching unit for a given vector, t
            Returns: bmu and bmu_idx is the index of this vector in the SOM
        """
        bmu_idx = np.array([0, 0])
        min_dist = np.Inf

        # calculate the distance between each neuron and the input
        for x in range(self.net.shape[0]):
            for y in range(self.net.shape[1]):
                w = self.net[x, y, :].reshape(self.m, 1)
                sq_dist = np.sum((w - t) ** 2)
                sq_dist = np.sqrt(sq_dist)
                if sq_dist < min_dist:
                    min_dist = sq_dist  # dist
                    bmu_idx = np.array([x, y])  # id

        bmu = self.net[bmu_idx[0], bmu_idx[1], :].reshape(self.m, 1)
        return bmu, bmu_idx

    def decay_radius(self, i):
        return self.init_radius * np.exp(-i / self.time_constant)

    def decay_learning_rate(self, i):
        return self.init_learning_rate * np.exp(-i / self.n_iterations)

    @staticmethod
    def calculate_influence(distance, radius):
        return np.exp(-distance / (2 * (radius ** 2)))

    def plot(self):
        fig = plt.figure()

        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.net.shape[0] + 1))
        ax.set_ylim((0, self.net.shape[1] + 1))
        ax.set_title('Self-Organising Map after %d iterations' % self.n_iterations)

        # plot
        for x in range(1, self.net.shape[0] + 1):
            for y in range(1, self.net.shape[1] + 1):
                ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                               facecolor=self.net[x - 1, y - 1],
                                               edgecolor='none'))
        plt.show()

