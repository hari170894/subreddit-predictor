#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
from gensim.corpora.dictionary import Dictionary
import pandas as pd
import logistics_pickler


def apply_feature_representation(in_file, out_file):
    print("converting file to BOW format {} -> {}".format(in_file, out_file))

    print("    reading csv")
    df = pd.read_csv(in_file, header=0)

    # Create dict that maps from words to IDs
    print("    creating dictionary")
    train_documents = (x.split() for x in df['text'])
    mydict = Dictionary([["<UNK>"]])
    mydict.add_documents(train_documents)

    def word_is_known(word):
        return word in mydict.values()

    def filter_unknowns_in_doc(doc):
        return [word if word_is_known(word) else "<UNK>" for word in doc]

    print("    processing all documents")
    all_feature_vectors = []
    for doc in train_documents:
        doc = filter_unknowns_in_doc(doc)
        features = mydict.doc2bow(doc)
        all_feature_vectors.append(features)

    print("    saving file")
    logistics_pickler.save_obj(all_feature_vectors, out_file, print_debug_info=False)


def main():
    apply_feature_representation("../res/data_all.csv", "../pickle_files/feature_all.pkl")
    apply_feature_representation("../res/data_sample.csv", "../pickle_files/feature_sample.pkl")
    apply_feature_representation("../res/data_train_all.csv", "../pickle_files/feature_train_all.pkl")
    apply_feature_representation("../res/data_test_all.csv", "../pickle_files/feature_test_all.pkl")
    apply_feature_representation("../res/data_validate_all.csv", "../pickle_files/feature_validate_all.pkl")


if __name__ == "__main__":
    main()
