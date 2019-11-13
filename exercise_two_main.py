import argparse
from setup_data import *
from kmeans import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("-p", help="Algorithm", choices={"km", "aj", "k"}, type=str, default='km', required=False)
parser.add_argument("-s", help="Split testing and test data percentage.", type=float, default=0.9, required=False)
parser.add_argument("-k", help="KNN k", type=int, default=5, required=False)
parser.add_argument("-km", help="K from k-means", type=int, default=5, required=False)
args = parser.parse_args()

point = args.p
split = args.s
k = args.k
km = args.km
authors = ['Calderaro', 'Fonteveccia', 'VanderKooy', 'Verbitsky', 'Pagni']
attributes_to_consider = 'abcdefgh'

attributes, classifications = read_texts(authors, attributes_to_consider)
training_set, test_set, training_set_class, test_set_class = setup_training_test_sets_joined(attributes, classifications, split)
kmeans = KMeans(km, training_set)
kmeans.train()

for index, element in enumerate(test_set):
    print(kmeans.predict(element), test_set_class[index])
