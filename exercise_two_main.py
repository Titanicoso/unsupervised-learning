import argparse
from setup_data import *
from kmeans import KMeans
from kohonen import Kohonen

parser = argparse.ArgumentParser()
parser.add_argument("-p", help="Algorithm", choices={"km", "aj", "som"}, type=str, default='km', required=False)
parser.add_argument("-s", help="Split testing and test data percentage.", type=float, default=0.9, required=False)
parser.add_argument("-k", help="KNN k", type=int, default=5, required=False)
parser.add_argument("-km", help="K from k-means", type=int, default=5, required=False)
args = parser.parse_args()

point = args.p
split = args.s
k = args.k
km = args.km
authors = ['Calderaro', 'Fonteveccia', 'VanderKooy', 'Verbitsky', 'Pagni']
attributes_to_consider = 'abc'


def k_means(training_set, test_set, test_set_class):
    kmeans = KMeans(km, training_set)
    kmeans.train()

    for index, element in enumerate(test_set):
        print(kmeans.predict(element), test_set_class[index])


def kohonen(training_set, test_set, training_set_class, test_set_class):
    training_set_array = np.array(training_set).reshape(len(training_set[0]), len(training_set))
    kohonen = Kohonen(10000, 0.01, training_set_array)
    kohonen.train()
    # kohonen.plot()
    for index, element in enumerate(training_set):
        bmu, bmu_idx = kohonen.find_bmu(np.array(element).reshape(len(element), 1))
        print(bmu_idx, training_set_class[index])


attributes, classifications = read_texts(authors, attributes_to_consider)
training_set, test_set, training_set_class, test_set_class = setup_training_test_sets_joined(attributes, classifications, split)

if point == 'km':
    k_means(training_set, test_set, test_set_class)

elif point == 'som':
    kohonen(training_set, test_set, training_set_class, test_set_class)


