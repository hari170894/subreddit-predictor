#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
from gensim.corpora.dictionary import Dictionary
import pandas as pd
import logistics_pickler
from tqdm import tqdm
import numpy as np


def make_dictionary_and_known_words(training_filename):
    print("reading training csv")
    df = pd.read_csv(training_filename, header=0)

    # Create dict that maps from words to IDs
    print("creating dictionary")
    train_documents = (x.split() for x in df['text'])
    dictionary = Dictionary([["<UNK>"]])
    dictionary.add_documents(train_documents)

    known_words = set(dictionary.values())

    return dictionary, known_words


def doc2bow(doc, dictionary, known_words):
    def word_is_known(word):
        return word in known_words

    def filter_unknowns_in_doc(doc):
        return [word if word_is_known(word) else "<UNK>" for word in doc]

    doc = filter_unknowns_in_doc(doc)
    bag_of_words = dictionary.doc2bow(doc)
    return bag_of_words


def bow2vec(bag_of_words, known_words):
    # the last 2 slots are for <UNK> and <NEVER OCCURS>
    features_vector = np.zeros(len(known_words) + 2, dtype=np.uint16)
    for word_id, count in bag_of_words:
        features_vector[word_id] = count
    return features_vector


def test_csv_to_pickle(test_filename, pickle_filename, dictionary, known_words):
    X = []
    Y = []

    print("loading csv into memory")
    df = pd.read_csv(test_filename, header=0)
    num_rows = df.shape[0]  # sum(1 for line in open(test_filename))

    print("converting to feature representation")
    for i, row in tqdm(df.iterrows(), total=num_rows):
        doc = row['text'].split()
        subreddit = row['subreddit']

        bag_of_words = doc2bow(doc, dictionary, known_words)
        features_vector = bow2vec(bag_of_words, known_words)

        X.append(features_vector)
        Y.append(subreddit)

    print("converting to numpy arrays")
    X = np.array(X)
    Y = np.array(Y)

    logistics_pickler.save_obj((X, Y), pickle_filename)


def main():
    dictionary, known_words = make_dictionary_and_known_words("../res/data_all_train.csv")
    logistics_pickler.save_obj(dictionary, "feature_bow_dictionary.pkl")

    test_csv_to_pickle("../res/data_all_train.csv", "../pickle_files/feature_all_train_bow.pkl", dictionary, known_words)
    test_csv_to_pickle("../res/data_all_test.csv", "../pickle_files/feature_all_test_bow.pkl", dictionary, known_words)
    test_csv_to_pickle("../res/data_all_validate.csv", "../pickle_files/feature_all_validate_bow.pkl", dictionary, known_words)


if __name__ == "__main__":
    main()
