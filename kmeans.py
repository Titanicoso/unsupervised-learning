import random
import math
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches


class KMeans:

    def __init__(self, k, training_set):
        self.training_set = training_set
        self.centroids = {}
        self.k = k
        initial_classes = [random.randint(0, k - 1) for x in range(len(training_set))]
        self.calculate_centroids(initial_classes)

    def train(self):
        changed = True
        while changed:
            changed = self.calculate_centroids()

    def calculate_centroids(self, initial_classes=None):
        centroids = {}

        for position, element in enumerate(self.training_set):
            if initial_classes is not None:
                closest = initial_classes[position]
            else:
                closest = self.predict(element)
            centroid = centroids.get(closest)

            if centroid is None:
                centroids[closest] = (element, 1)
            else:
                centroids[closest] = (np.add(centroid[0], element), centroid[1] + 1)

        changed = False

        for index, centroid in centroids.items():
            new_centroid = np.divide(centroid[0], centroid[1])
            if self.centroids.get(index) is None or not np.array_equal(new_centroid, self.centroids[index]):
                changed = True
                self.centroids[index] = new_centroid

        return changed

    def predict(self, element):
        best_classification = None
        best_distance = 0
        for index, centroid in self.centroids.items():
            dist = self.euclidean_distance(element, centroid)
            if best_classification is None or dist < best_distance:
                best_classification = index
                best_distance = dist
        return best_classification

    def predictAll(self, elements):
        predictions = []
        for element in elements:
            predictions.append(self.predict(element))
        return predictions

    def plot(self, test_set, test_set_class):
        fig = plt.figure()

        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.k))
        ax.set_ylim((0, 1))
        ax.set_title("Clasificación")

        classes_color = {}
        colors = ['r', 'g', 'b', 'c', 'm']
        counter = 0
        classes = {}

        for index, element in enumerate(test_set):
            prediction = self.predict(element)
            classification = test_set_class[index]
            l = classes.get(prediction, [])
            l.append(classification)
            classes[prediction] = l

            if classes_color.get(classification) is None:
                classes_color[classification] = colors[counter]
                counter += 1

        for x in range(self.k):
            y = 0
            ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='white', edgecolor='black'))

            for classification in classes.get(x, []):
                x_aux = x + 0.5 + np.random.normal(0, 0.15)
                y_aux = y + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x_aux, y_aux, marker='.', color=classes_color[classification], markersize=24,
                         label=classification)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=[1, 0.5])
        plt.savefig("plots/class_kmeans.png", bbox_inches='tight')

    @staticmethod
    def euclidean_distance(element1, element2):

        squared_distance = 0

        for i in range(len(element1)):
            squared_distance += (element1[i] - element2[i]) ** 2

        return math.sqrt(squared_distance)
