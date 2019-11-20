import argparse
import math

import matplotlib.pyplot as plt
from setup_data import *
from kmeans import KMeans
from kohonen import Kohonen
from hierarchical_clustering import HierarchicalClustering
from scipy.cluster.hierarchy import dendrogram

parser = argparse.ArgumentParser()
parser.add_argument("-p", help="Algorithm", choices={"km", "hc", "som"}, type=str, default='hc', required=False)
parser.add_argument("-s", help="Split testing and test data percentage.", type=float, default=0.8, required=False)
parser.add_argument("-k", help="KNN k", type=int, default=5, required=False)
parser.add_argument("-attrs", help="Attributes to consider (abcdefg)", type=str, default='abcdefg', required=False)
parser.add_argument("-a", help="Authors to consider (c,f,p,va,ve)", type=str, default='c,f,p,va,ve', required=False)
parser.add_argument("-size", help="Size of network", type=int, required=False)
parser.add_argument("-km", help="K from k-means", type=int, default=5, required=False)
parser.add_argument("-l", help="Linkage", type=str, default='single', choices={'single', 'complete', 'average'}, required=False)
parser.add_argument("-n", help="Normalize", default=True, required=False, action='store_true')
parser.add_argument("-tr", help="Use training set for graphs.", default=False, required=False, action='store_true')
args = parser.parse_args()

point = args.p
split = args.s
k = args.k
km = args.km
size = args.size
linkage = args.l
attributes_to_consider = args.attrs
normalize = args.n
use_training = args.tr

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

    if use_training:
        array = training_set
        class_array = training_set_class
    else:
        array = test_set
        class_array = test_set_class

    kmeans.plot(array, class_array)

    for index, element in enumerate(array):
        print(kmeans.predict(element), class_array[index])


def kohonen(training_set, test_set, test_set_class):
    training_set_array = np.array(training_set).T
    kohonen = Kohonen(10000, 0.01, training_set_array, size, size)
    kohonen.train()

    if use_training:
        kohonen.plot(training_set, training_set_class)
        array = training_set
        class_array = training_set_class
    else:
        kohonen.plot(test_set, test_set_class)
        array = test_set
        class_array = test_set_class

    for index, element in enumerate(array):
        t = np.array(element).reshape(np.array([kohonen.m, 1]))
        bmu, bmu_idx = kohonen.find_bmu(t)
        print(bmu_idx, class_array[index])


def hierarchical_clustering(training_set):
    training_set_array = np.array(training_set)
    hc = HierarchicalClustering(training_set_array, linkage)
    z = hc.cluster()
    dendrogram(z, labels=training_set_class, link_color_func=lambda k: 'b')
    plt.savefig("plots/dendrogram.png", bbox_inches='tight')


attributes, classifications = read_texts(authors, attributes_to_consider)
training_set, test_set, training_set_class, test_set_class = setup_training_test_sets_joined(attributes, classifications,
                                                                                             split)
if normalize:
    attributes_group = [[] for i in range(len(attributes[0]))]

    for i, attribute in enumerate(attributes):
        for j in range(len(attribute)):
            attributes_group[j].append(attribute[j])

    normalized_attributes = [[] for i in range(len(attributes[0]))]
    for i, attribute_type in enumerate(attributes_group):
        max = np.max(attribute_type)
        for attribute in attribute_type:
            normalized_attributes[i].append(attribute / max)

    result = [[] for i in range(len(attributes))]
    for i, a in enumerate(attributes):
        for j in range(len(attributes[0])):
            result[i].append(normalized_attributes[j][i])

    training_set, test_set, training_set_class, test_set_class = setup_training_test_sets_joined(result, classifications, split)


if point == 'km':
    k_means(training_set, test_set, test_set_class)

elif point == 'som':
    kohonen(training_set, test_set, test_set_class)

else:
    hierarchical_clustering(training_set)


