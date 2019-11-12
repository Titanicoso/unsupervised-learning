import random
import numpy as np
import csv


def read_acath(add_sex=False):
    attributes = []
    classifications = []
    cholestes = []

    with open('./data/acath.csv', 'r') as csv_file:
        lines = csv.reader(csv_file, delimiter=';')
        dataset = list(lines)[1:]

        for line in dataset:
            sex = int(line[0])
            age = float(line[1])
            duration = float(line[2])
            choleste = line[3]
            sigdz = int(line[4])

            if choleste != '':
                cholestes.append(float(choleste))
                choleste = float(choleste)
            else:
                choleste = -1

            data_in_line = [age, duration, choleste]
            if add_sex:
                data_in_line.append(sex)

            attributes.append(data_in_line)
            classifications.append(sigdz)

    mean_choleste = int(np.average(cholestes))

    for attribute in attributes:
        if attribute[2] == -1:
            attribute[2] = mean_choleste

    return attributes, classifications


def setup_training_test_sets_joined(data, split):
    test_set = []
    training_set = data.copy()
    testing = random.sample(range(0, len(training_set) - 1), int(len(training_set) * (1 - split)))
    for index in sorted(testing, reverse=True):
        test_set.append(training_set.pop(index))
    return training_set, test_set

def setup_training_test_sets(x, y, split):
    test_set_y = []
    training_set_y = y.copy()

    test_set_X = []
    training_set_X = x.copy()

    testing_indexes = random.sample(range(0, len(training_set_X) - 1), int(len(training_set_X) * (1 - split)))

    for index in sorted(testing_indexes, reverse=True):
        test_set_X.append(training_set_X.pop(index))
        test_set_y.append(training_set_y.pop(index))

    return training_set_X, test_set_X, training_set_y, test_set_y

def setup_data_logistic(attributes, classification, split):
    training_set_X, test_set_X, training_set_y, test_set_y = setup_training_test_sets(attributes, classification, split)

    training_set_X = np.matrix(training_set_X, dtype=np.float32)
    test_set_X = np.matrix(test_set_X, dtype=np.float32)
    training_set_y = np.array(training_set_y, dtype=np.int).ravel()
    test_set_y = np.array(test_set_y, dtype=np.int).ravel()

    return training_set_X, test_set_X, training_set_y, test_set_y
