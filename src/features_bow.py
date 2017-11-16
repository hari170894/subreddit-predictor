#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
from gensim.corpora.dictionary import Dictionary
import pandas as pd
import logistics_pickler
from tqdm import tqdm
import numpy as np


def apply_feature_representation(in_file, out_file):
    print("converting file to BOW format {} -> {}".format(in_file, out_file))

    print("    reading csv")
    df = pd.read_csv(in_file, header=0)

    # Create dict that maps from words to IDs
    print("    creating dictionary")
    train_documents = (x.split() for x in df['text'])
    mydict = Dictionary([["<UNK>"]])
    mydict.add_documents(train_documents)

    known_words = set(mydict.values())

    def word_is_known(word):
        return word in known_words

    def filter_unknowns_in_doc(doc):
        return [word if word_is_known(word) else "<UNK>" for word in doc]

    X = []
    Y = []

    print("    counting number of rows")
    num_rows = sum(1 for line in open(in_file))

    print("    converting to feature representation")
    for i, row in tqdm(df.iterrows(), total=num_rows):
        doc = row['text'].split()
        subreddit = row['subreddit']

        doc = filter_unknowns_in_doc(doc)
        features_bow = mydict.doc2bow(doc)

        # the last 2 slots are for <UNK> and <NEVER OCCURS>
        features_vector = np.zeros(len(known_words) + 2, dtype=np.uint16)
        for word_id, count in features_bow:
            features_vector[word_id] = count

        X.append(features_vector)
        Y.append(subreddit)

    print("    converting to numpy arrays")
    X = np.array(X)
    Y = np.array(Y)

    print("    saving file")
    logistics_pickler.save_obj((X, Y), out_file, print_debug_info=False)


def main():
    apply_feature_representation("../res/data_sample.csv", "../pickle_files/feature_sample_bow.pkl")
    apply_feature_representation("../res/data_all.csv", "../pickle_files/feature_all_bow.pkl")
    apply_feature_representation("../res/data_all_train.csv", "../pickle_files/feature_all_train_bow.pkl")
    apply_feature_representation("../res/data_all_test.csv", "../pickle_files/feature_all_test_bow.pkl")
    apply_feature_representation("../res/data_all_validate.csv", "../pickle_files/feature_all_validate_bow.pkl")


if __name__ == "__main__":
    main()
