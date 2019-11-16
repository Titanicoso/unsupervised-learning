import random
import numpy as np
import csv
import os
from collections import Counter
import collections


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


def read_texts(authors, attributes_to_consider):
    attributes = []
    classifications = []

    for author in authors:
        attributes_aux, classifications_aux = read_texts_author(author, attributes_to_consider)
        attributes.extend(attributes_aux)
        classifications.extend(classifications_aux)

    return attributes, classifications


def get_words(lines):
    words = []
    words_in_phrase_count = []
    phrases = []

    for line in lines:
        for phrase in line.split('.'):
            if phrase != '\n':
                phrases.append(phrase)
                words_in_phrase = phrase.split()
                words_in_phrase_count.append(len(words_in_phrase))
                words.extend([word.replace('.', '').replace(',', '') for word in words_in_phrase])

    return words, round(np.mean(words_in_phrase_count)/len(words), 4)


def get_articles_frequencies(distinct_words, total_word_count):
    deterministic_articles = ["la", "el", "los", "las"]
    nondeterministic_articles = ["un", "una", "unos", "unas"]
    deterministic_article_count = 0
    nondeterministic_article_count = 0

    for article in deterministic_articles:
        deterministic_article_count += distinct_words.get(article, 0)

    for article in nondeterministic_articles:
        nondeterministic_article_count += distinct_words.get(article, 0)

    return round(deterministic_article_count/total_word_count, 3), round(nondeterministic_article_count/total_word_count, 3)


def get_distinct_words(words):
    distinct_words = Counter(words)
    distinct_words = sorted(distinct_words.items(), key=lambda x: x[1], reverse=True)
    distinct_words = collections.OrderedDict(distinct_words)
    return distinct_words


def get_sum_frequency(amount_different_words, distinct_words):
    # TODO: cantidad de palabras totales o distintas?
    return float(np.sum([frequency/amount_different_words for frequency in list(distinct_words.values())[:5]]))


def get_conjuctions(text):
    subordinant_conjuctions = ["porque", "pues", "ya que", "puesto que", "como", "que", "aunque", "aun cuando",
                               "si bien"
                               "a causa", "de", "debido a", "luego", "conque", "así que", "si", "para que",
                               "a fin de que"]
    coordinant_conjuctions = ["ni", "y", "o", "o bien", "pero aunque", "no obstante", "sin embargo", "sino",
                              "por el contrario"]

    subordinant_conjuctions_count = 0
    coordinant_conjuctions_count = 0

    text = text.lower()

    for conjunction in subordinant_conjuctions:
        subordinant_conjuctions_count += text.count(conjunction)

    for conjunction in coordinant_conjuctions:
        coordinant_conjuctions_count += text.count(conjunction)

    return subordinant_conjuctions_count, coordinant_conjuctions_count


def read_texts_author(author, attributes_to_consider):
    file_names = []
    attributes = []
    classifications = []

    for filename in os.listdir(author):
        file_names.append(os.path.join(author, filename))

    for file_name in file_names:
        attribute = []

        with open(file_name, 'r', encoding='utf-8') as f:
            lines = [line.lower() for line in f]
            words, mean_word_count_in_line = get_words(lines)
            total_word_count = len(words)

            distinct_words = get_distinct_words(words)
            deterministic_article_count, nondeterministic_article_count = get_articles_frequencies(distinct_words, total_word_count)
            distinct_words_count = round(len(distinct_words) / total_word_count, 3)
            sum_frequency = round(get_sum_frequency(len(distinct_words), distinct_words), 3)
            subordinant_conjuctions_count, coordinant_conjuctions_count = get_conjuctions(" ".join(lines))

            if 'a' in attributes_to_consider:
                attribute.append(mean_word_count_in_line)

            if 'b' in attributes_to_consider:
                attribute.append(sum_frequency)

            if 'c' in attributes_to_consider:
                attribute.append(distinct_words_count)

            if 'd' in attributes_to_consider:
                attribute.append(subordinant_conjuctions_count)

            if 'e' in attributes_to_consider:
                attribute.append(coordinant_conjuctions_count)

            if 'f' in attributes_to_consider:
                attribute.append(deterministic_article_count)

            if 'g' in attributes_to_consider:
                attribute.append(nondeterministic_article_count)

            attributes.append(attribute)
            classifications.append(author)

    return attributes, classifications


def setup_training_test_sets_joined(data, classifications, split):
    test_set = []
    test_set_class = []
    training_set = data.copy()
    training_set_class = classifications.copy()

    testing = random.sample(range(0, len(training_set) - 1), int(len(training_set) * (1 - split)))

    for index in sorted(testing, reverse=True):
        test_set.append(training_set.pop(index))
        test_set_class.append(training_set_class.pop(index))

    return training_set, test_set, training_set_class, test_set_class


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
