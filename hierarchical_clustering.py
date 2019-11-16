import numpy as np


class Cluster:

    def __init__(self, elements, index):
        self.elements = elements
        self.index = index

    def distance(self, other_cluster, distance_matrix, linkage):
        if linkage is 'single':
            return self.min_distance(other_cluster, distance_matrix)
        elif linkage is 'complete':
            return self.max_distance(other_cluster, distance_matrix)
        elif linkage is 'average':
            return self.avg_distance(other_cluster, distance_matrix)
        else:
            return self.min_distance(other_cluster, distance_matrix)

    def max_distance(self, other_cluster, distance_matrix):
        distance = -1

        for element in self.elements:
            for other_element in other_cluster.elements:
                dist = distance_matrix[element][other_element]
                if dist > distance:
                    distance = dist

        return distance

    def min_distance(self, other_cluster, distance_matrix):
        distance = None

        for element in self.elements:
            for other_element in other_cluster.elements:
                dist = distance_matrix[element][other_element]
                if distance is None or dist < distance:
                    distance = dist

        return distance

    def avg_distance(self, other_cluster, distance_matrix):
        distance = 0

        for element in self.elements:
            for other_element in other_cluster.elements:
                distance += distance_matrix[element][other_element]

        return distance / (len(self.elements) * len(other_cluster.elements))


class HierarchicalClustering:

    def __init__(self, training_set, linkage='single'):
        self.training_set = training_set
        self.normalize()
        self.linkage = linkage
        self.distance_matrix = np.zeros((len(training_set), len(training_set)))
        self.clusters = [Cluster([index], index) for index, element in enumerate(training_set)]
        self.linkage_matrix = np.zeros((len(training_set) - 1, 4))
        self.last_index = len(training_set) - 1
        self.calculate_initial_distances()

    def calculate_initial_distances(self):
        for index, element in enumerate(self.training_set):
            for other_index, other_element in enumerate(self.training_set):
                if index != other_index:
                    dist = np.linalg.norm(element - other_element)
                    self.distance_matrix[index][other_index] = dist

    def cluster(self):
        iteration = 0
        while len(self.clusters) != 1:
            distance = None
            cluster1 = None
            cluster2 = None

            for cluster in self.clusters:
                for other_cluster in self.clusters:
                    if other_cluster != cluster:
                        dist = cluster.distance(other_cluster, self.distance_matrix, self.linkage)
                        if distance is None or dist < distance:
                            cluster1 = cluster
                            cluster2 = other_cluster
                            distance = dist

            self.clusters.remove(cluster1)
            self.clusters.remove(cluster2)
            self.clusters.append(Cluster(cluster1.elements + cluster2.elements, self.last_index + 1))
            self.last_index += 1

            self.linkage_matrix[iteration][0] = cluster1.index
            self.linkage_matrix[iteration][1] = cluster2.index
            self.linkage_matrix[iteration][2] = distance
            self.linkage_matrix[iteration][3] = len(cluster1.elements) + len(cluster2.elements)

            iteration += 1
        return self.linkage_matrix

    def normalize(self):
        max_elements = np.max(self.training_set, axis=0)
        self.training_set /= max_elements

