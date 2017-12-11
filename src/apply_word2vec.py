#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
# from gensim.corpora.dictionary import Dictionary
import pandas as pd
import logistics_pickler
from tqdm import tqdm
import numpy as np
import nltk


'''
load glove.840B.300d.txt into dict

for each file
    for each example
        for each word
            apply model
            if unknown, make it 100 zeros
    write that to a .pkl in pickle_files

'''

from collections import defaultdict

# glove_input_filename = '../res/glove.840B.300d.txt'
# glove_output_filename = '../pickle_files/word2vec_glove.pkl'
# import os.path
# if os.path.isfile(glove_output_filename):
#     print("is file")
# else:
#     print("isn't file")
# exit()

def load_model():
    glove_input_filename = '../res/glove.840B.300d.txt'
    glove_output_filename = '../pickle_files/word2vec_glove.pkl'
    # glove_filename = '../res/glove.840B.300d.txt'
    import os.path
    if not os.path.isfile(glove_output_filename):
        word2vec_dict = {}
        word2vec_dict = defaultdict(lambda: [0] * 100, word2vec_dict)

        print("preparing to load word2vec file")
        with open(glove_input_filename) as f:
            for line in tqdm(f, total=2196017):
                love = line.split()
                word, nums = love[0], [float(x) for x in love[1:]]
                word2vec_dict[word] = nums

        logistics_pickler.save_obj(word2vec_dict, 'word2vec_glove')
    else:
        word2vec_dict = logistics_pickler.load_obj('word2vec_glove')
    return word2vec_dict


def vectorize_words(words, model):
    return [model[word] for word in words]


# train_df = pd.read_csv('../res/data_all_train.csv', header=0)
# test_df = pd.read_csv('../res/data_all_test.csv', header=0)
# validation_df = pd.read_csv('../res/data_all_validate.csv', header=0)


def sent_vectorizer_glove(sent, model):
    sent_vec = np.zeros(300)
    for word in nltk.word_tokenize(sent.decode('utf8')):
        if word in model.wv.vocab:
            sent_vec = np.add(sent_vec, word2vec_dict[word])
    return sent_vec


def create_features():
    train_df = pd.read_csv('../res/data_all_train.csv', header=0)
    test_df = pd.read_csv('../res/data_all_test.csv', header=0)
    validation_df = pd.read_csv('../res/data_all_validate.csv', header=0)
    documents = []
    for doc in train_df.text:
        unicode_string = nltk.word_tokenize(doc.decode('utf8'))
        documents.append(unicode_string)

    print(len(documents))

    # model = Word2Vec(documents, size=300)
    model = load_model()

    # model.train(documents, total_examples=len(documents), epochs=20)

    create_and_save_from_model(model, train_df, "features_train_word2vec_glove")
    create_and_save_from_model(model, test_df, "features_test_word2vec_glove")
    create_and_save_from_model(model, validation_df, "features_validation_word2vec_glove")


def create_and_save_from_model(model, df, filename):
    X = []
    Y = []
    T = []
    # values_to_write = []
    for index, row in df.iterrows():
        if modeltype == 0:
            features_vector = vectorize_words(row.text, model)
        elif modeltype == 1:
            features_vector = sent_vectorizer_maximum(row.text, model)
        label = row.subreddit
        # values_to_write.append([features_vector, label])
        T.append(row.text)
        X.append(features_vector)
        Y.append(label)
    # save_obj(values_to_write, filename)
    T = np.array(T)
    X = np.array(X)
    Y = np.array(Y)

    logistics_pickler.save_obj((T, X, Y), filename)


def main():
    load_model()
    # pass


if __name__ == "__main__":
    main()
