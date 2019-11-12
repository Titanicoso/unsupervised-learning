import argparse

from knn import KNN
from setup_data import *
from sklearn.linear_model import LogisticRegression
from kmeans import KMeans

def train_logistic_regressor(attributes, classification, split):
    training_set_X, test_set_X, training_set_y, test_set_y = setup_data_logistic(attributes, classification, split)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(training_set_X, training_set_y)
    return lr, training_set_X, test_set_X, training_set_y, test_set_y

def point_b():
    lr, training_set_X, test_set_X, training_set_y, test_set_y = \
        train_logistic_regressor(attributes, classification, split)
    predicted = lr.predict(test_set_X)
    return
    # TODO: confusion matrix

def point_c():
    lr, training_set_X, test_set_X, training_set_y, test_set_y = \
        train_logistic_regressor(attributes, classification, split)
    prediction = lr.predict_proba(np.matrix([60, 2, 199], dtype=np.float32))
    print(prediction)
    return

def point_d():
    attributes, classification = read_acath(True)
    lr, training_set_X, test_set_X, training_set_y, test_set_y = \
        train_logistic_regressor(attributes, classification, split)
    predicted = lr.predict(test_set_X)
    return
    #TODO: confusion matrix

def point_e():
    attributes, classification = read_acath()

    for index, data in enumerate(attributes):
        data.append(classification[index])

    training_set, test_set = setup_training_test_sets_joined(attributes, split)
    knn = KNN(k, False, training_set, test_set)
    knn.run()
    return
    # TODO: confusion matrix

def point_f():
    attributes, classification = read_acath()
    training_set, test_set = setup_training_test_sets_joined(attributes, split)
    kmeans = KMeans(km, training_set)
    kmeans.train()
    print(kmeans.predict([73, 132, 268]))
    print(kmeans.predict([58, 86, 245]))
    return

parser = argparse.ArgumentParser()
parser.add_argument("-p", help="Point", choices={"b", "c", "d", "e", "f"}, type=str, default='f', required=False)
parser.add_argument("-s", help="Split testing and test data percentage.", type=float, default=0.9, required=False)
parser.add_argument("-k", help="KNN k", type=int, default=5, required=False)
parser.add_argument("-km", help="K from k-means", type=int, default=2, required=False)
args = parser.parse_args()

point = args.p
split = args.s
k = args.k
km = args.km

attributes, classification = read_acath()

if point == "b":
    point_b()
elif point == "c":
    point_c()
elif point == "d":
    point_d()
elif point == "e":
    point_e()
else:
    point_f()

