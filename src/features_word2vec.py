from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
from gensim.models import Word2Vec
import pandas as pd
import nltk
import numpy as np
from logistics_pickler import save_obj


def sent_vectorizer_addition(sent, model):
    sent_vec = np.zeros(300)
    for word in nltk.word_tokenize(sent.decode('utf8')):
        if word in model.wv.vocab:
            sent_vec = np.add(sent_vec, model[word])
    return sent_vec


def sent_vectorizer_maximum(sent, model):
    sent_vec = np.zeros(300)
    for word in nltk.word_tokenize(sent.decode('utf8')):
        if word in model.wv.vocab:
            sent_vec = np.maximum(sent_vec, model[word])
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

    model = Word2Vec(documents, size=300)

    model.train(documents, total_examples=len(documents), epochs=20)

    create_and_save_from_model(model, train_df, "features_train_word2vec_sum", 0)
    create_and_save_from_model(model, test_df, "features_test_word2vec_sum", 0)
    create_and_save_from_model(model, validation_df, "features_validation_word2vec_sum", 0)

    create_and_save_from_model(model, train_df, "features_train_word2vec_max", 1)
    create_and_save_from_model(model, test_df, "features_test_word2vec_max", 1)
    create_and_save_from_model(model, validation_df, "features_validation_word2vec_max", 1)


def create_and_save_from_model(model, df, filename, modeltype):
    values_to_write = []
    for index, row in df.iterrows():
        if modeltype == 0:
            vector = sent_vectorizer_addition(row.text, model)
        elif modeltype == 1:
            vector = sent_vectorizer_maximum(row.text, model)
        label = row.subreddit
        values_to_write.append([vector, label])
    save_obj(values_to_write, filename)


create_features()
