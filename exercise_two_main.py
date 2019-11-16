import argparse
import math

from setup_data import *
from kmeans import KMeans
from kohonen import Kohonen

parser = argparse.ArgumentParser()
parser.add_argument("-p", help="Algorithm", choices={"km", "aj", "som"}, type=str, default='km', required=False)
parser.add_argument("-s", help="Split testing and test data percentage.", type=float, default=0.8, required=False)
parser.add_argument("-k", help="KNN k", type=int, default=5, required=False)
parser.add_argument("-attrs", help="Attributes to consider", type=str, default='abcdefg', required=False)
parser.add_argument("-a", help="Authors to consider", type=str, default='c,f,p,va,ve', required=False)
parser.add_argument("-size", help="Size of network", type=int, required=False)
parser.add_argument("-km", help="K from k-means", type=int, default=5, required=False)
args = parser.parse_args()

point = args.p
split = args.s
k = args.k
km = args.km
size = args.size
attributes_to_consider = args.attrs

authors = []
if 'c' in args.a:
    authors.append('Calderaro')

if 'f' in args.a:
    authors.append('Fonteveccia')

if 'va' in args.a:
    authors.append('VanderKooy')

if 've' in args.a:
    authors.append('Verbitsky')

if 'p' in args.a:
    authors.append('Pagni')

if size is None:
    size = int(np.ceil(math.log2(len(authors))))


def k_means(training_set, test_set, test_set_class):
    kmeans = KMeans(km, training_set)
    kmeans.train()
    kmeans.plot(test_set, test_set_class)

    for index, element in enumerate(test_set):
        print(kmeans.predict(element), test_set_class[index])


def kohonen(training_set, test_set, test_set_class):
    training_set_array = np.array(training_set).T
    kohonen = Kohonen(10000, 0.01, training_set_array, size, size)
    kohonen.train()
    kohonen.plot(test_set, test_set_class)

    for index, element in enumerate(test_set):
        t = np.array(element).reshape(np.array([kohonen.m, 1]))
        bmu, bmu_idx = kohonen.find_bmu(t)
        print(bmu_idx, test_set_class[index])


attributes, classifications = read_texts(authors, attributes_to_consider)
training_set, test_set, training_set_class, test_set_class = setup_training_test_sets_joined(attributes, classifications, split)

if point == 'km':
    k_means(training_set, test_set, test_set_class)

elif point == 'som':
    kohonen(training_set, test_set, test_set_class)


