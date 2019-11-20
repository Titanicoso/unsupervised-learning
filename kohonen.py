from collections import defaultdict, OrderedDict
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches


class BmuIndx:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __eq__(self, o) -> bool:
        return o.x == self.x and o.y == self.y

    def __str__(self) -> str:
        return str(self.x) + " " + str(self.y)

    def __hash__(self) -> int:
        return hash(self.x) + hash(self.y)


class Kohonen:

    def __init__(self, n_iterations, init_learning_rate, training_set, m, n) -> None:

        network_dimensions = np.array([m, n])
        self.n_iterations = n_iterations
        self.init_learning_rate = init_learning_rate

        self.m = training_set.shape[0]
        self.n = training_set.shape[1]

        # initial neighbourhood radius
        self.init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
        # radius decay parameter
        self.time_constant = self.n_iterations / np.log(self.init_radius)

        self.data = training_set

        normalise_data = True

        if normalise_data:
            self.data = training_set / training_set.max(axis=1)[:, np.newaxis]

        self.net = np.random.random((network_dimensions[0], network_dimensions[1], self.m))

        for x in range(self.net.shape[0]):
            for y in range(self.net.shape[1]):
                self.net[x, y, :] = self.data[:, np.random.randint(0, self.n)]

    def train(self):
        for i in range(self.n_iterations):

            t = self.data[:, np.random.randint(0, self.n)].reshape(np.array([self.m, 1]))
            bmu, bmu_idx = self.find_bmu(t)

            # ajustar parametros
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
                        # calcular influencia (basado en distancia 2d)
                        influence = Kohonen.calculate_influence(w_dist, r)

                        # new w = old w + (learning rate * influence * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (l * influence * (t - w))
                        self.net[x, y, :] = new_w.reshape(1, len(t))

    # best matching unit
    def find_bmu(self, t):
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

    # funcion de vecinidad
    @staticmethod
    def calculate_influence(distance, radius):
        return np.exp(-distance / (2 * (radius ** 2)))

    def plot(self, test_set, test_set_class):
        bmus = defaultdict(list)
        classes_color = {}
        colors = ['r', 'g', 'b', 'c', 'm']
        counter = 0

        for index, element in enumerate(test_set):
            t = np.array(element).reshape(np.array([self.m, 1]))
            bmu, bmu_idx = self.find_bmu(t)

            bmuIndex = BmuIndx(bmu_idx[0], bmu_idx[1])

            classification = test_set_class[index]
            l = bmus.get(bmuIndex, [])
            l.append(classification)
            bmus[bmuIndex] = l

            if classes_color.get(classification) is None:
                classes_color[classification] = colors[counter]
                counter += 1

        self.plot_markers(classes_color, bmus)

        for classification in classes_color.keys():
            self.plot_class_density(bmus, classification, len(test_set))

    def plot_class_density(self, bmus, classification, max):
        fig = plt.figure()

        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.net.shape[0]))
        ax.set_ylim((0, self.net.shape[1]))
        ax.set_title("Densidad para " + classification)

        for x in range(self.net.shape[0]):
            for y in range(self.net.shape[1]):
                index = BmuIndx(x, y)
                color = bmus.get(index, []).count(classification)

                if color != 0:
                    aux = color / max
                    color = (1-aux, aux, aux)
                else:
                    color = (1, 1, 1)

                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black'))

        plt.savefig("plots/densidad" + classification + ".png", bbox_inches='tight')
        # plt.show()

    def plot_markers(self, classes_color, bmus):
        fig = plt.figure()

        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.net.shape[0]))
        ax.set_ylim((0, self.net.shape[1]))
        ax.set_title("Clasificación")

        for x in range(self.net.shape[0]):
            for y in range(self.net.shape[1]):
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='white', edgecolor='black'))

                for classification in bmus.get(BmuIndx(x, y), []):
                    x_aux = x + 0.5 + np.random.normal(0, 0.1)
                    y_aux = y + 0.5 + np.random.normal(0, 0.1)
                    plt.plot(x_aux, y_aux, marker='.', color=classes_color[classification], markersize=24, label=classification)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=[1, 0.5])
        plt.savefig("plots/class_kohonen.png", bbox_inches='tight')
        # plt.show()
