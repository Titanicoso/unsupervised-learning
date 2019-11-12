import random
import math
import numpy as np


class KMeans:

    def __init__(self, k, training_set):
        self.training_set = training_set
        self.centroids = {}
        self.k = k
        initial_classes = [random.randint(1, k) for x in range(len(training_set))]
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
        for index, centroid in self.centroids.items():
            dist = self.euclidean_distance(element, centroid)
            if best_classification is None or dist < best_classification:
                best_classification = index
        return best_classification

    @staticmethod
    def euclidean_distance(element1, element2):

        squared_distance = 0

        for i in range(len(element1)):
            squared_distance += (element1[i] - element2[i]) ** 2

        return math.sqrt(squared_distance)
