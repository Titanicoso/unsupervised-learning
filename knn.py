import math
import operator

class KNN:

    def __init__(self, k, is_weighed, training_set, test_set):
        self.k = k
        self.is_weighed = is_weighed
        self.training_set = training_set
        self.test_set = test_set
        self.predictions = []

    def run(self):
        for test_instance in self.test_set:
            neighbors = self.get_neighbors(test_instance)
            result = self.get_response(test_instance, neighbors)
            self.predictions.append(result)

        return self.predictions

    def get_response(self, test_instance, neighbors):
        class_votes = {}

        for x in range(len(neighbors)):

            votes = neighbors[x][-1]
            result = 1

            if self.is_weighed:
                distance = self.euclidean_distance(test_instance, neighbors[x], len(test_instance) - 1) ** 2
                if distance == 0:
                    return votes
                else:
                    result = 1 / distance

            if votes in class_votes:
                class_votes[votes] += result
            else:
                class_votes[votes] = result

        return max(class_votes.items(), key=operator.itemgetter(1))[0]

    def get_neighbors(self, test_instance):
        distances = []
        length = len(test_instance) - 1

        for x in range(len(self.training_set)):
            dist = self.euclidean_distance(test_instance, self.training_set[x], length)
            distances.append((self.training_set[x], dist))

        distances.sort(key=operator.itemgetter(1))
        neighbors = []

        for x in range(self.k):
            neighbors.append(distances[x][0])

        return neighbors

    @staticmethod
    def euclidean_distance(instance1, instance2, length):
        distance = 0

        for x in range(length):
            distance += (instance1[x] - instance2[x]) ** 2

        return math.sqrt(distance)
