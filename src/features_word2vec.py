from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
from gensim.models import Word2Vec
import pandas as pd
import nltk
import numpy as np
from logistics_pickler import save_obj, load_obj
from tqdm import tqdm


def load_model():
    glove_input_filename = '../res/glove.twitter.27B.100d.txt'
    glove_output_filename = '../pickle_files/word2vec_glove.pkl'
    # glove_filename = '../res/glove.840B.300d.txt'
    import os.path
    if not os.path.isfile(glove_output_filename) or True:
        # word2vec_dict = {}
        # word2vec_dict = defaultdict(lambda: [0] * 100, word2vec_dict)
        word2vec_dict = dict()

        with open(glove_input_filename) as f:
            for line in tqdm(f, total=1193514):
                love = line.split()
                word, nums = love[0], [float(x) for x in love[1:]]
                word2vec_dict[word] = nums

        # print("saving representation")
        # save_obj(word2vec_dict, 'word2vec_glove')
        # print("representation saved")
    else:
        print("loading word2vec glove feature representation")
        word2vec_dict = load_obj('word2vec_glove')
        print("representation loaded")
    return word2vec_dict


def sent_vectorizer_addition(sent, model):
    sent_vec = np.zeros(100)
    for word in nltk.word_tokenize(sent.decode('utf8')):
        # if word in model.wv.vocab:
        if word in model:
            sent_vec = np.add(sent_vec, model[word])
    return sent_vec


def sent_vectorizer_maximum(sent, model):
    sent_vec = np.zeros(100)
    for word in nltk.word_tokenize(sent.decode('utf8')):
        # if word in model.wv.vocab:
        if word in model:
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

    print("number of documents: {}".format(len(documents)))

    # model = Word2Vec(documents, size=300)
    # model.train(documents, total_examples=len(documents), epochs=20)
    model = load_model()

    create_and_save_from_model(model, train_df, "features_train_word2vec_sum", 0)
    create_and_save_from_model(model, test_df, "features_test_word2vec_sum", 0)
    create_and_save_from_model(model, validation_df, "features_validation_word2vec_sum", 0)

    create_and_save_from_model(model, train_df, "features_train_word2vec_max", 1)
    create_and_save_from_model(model, test_df, "features_test_word2vec_max", 1)
    create_and_save_from_model(model, validation_df, "features_validation_word2vec_max", 1)


def create_and_save_from_model(model, df, filename, modeltype):
    print("applying features to {}".format(filename))
    X = []
    Y = []
    T = []
    # values_to_write = []
    for index, row in df.iterrows():
        if modeltype == 0:
            features_vector = sent_vectorizer_addition(row.text, model)
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

    print("saving features for {}".format(filename))
    save_obj((T, X, Y), filename)


def main():
    create_features()


if __name__ == "__main__":
    main()
