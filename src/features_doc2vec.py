from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import pandas as pd
import nltk
import numpy as np
from logistics_pickler import save_obj


def create_features():
    train_df = pd.read_csv('../res/data_all_train.csv', header=0)
    test_df = pd.read_csv('../res/data_all_test.csv', header=0)
    validation_df = pd.read_csv('../res/data_all_validate.csv', header=0)
    i = 1
    documents = []
    for doc in train_df.text:
        unicode_string = nltk.word_tokenize(doc.decode('utf8'))
        tagged_document = TaggedDocument(unicode_string, str("D" + str(i)))
        i += 1
        documents.append(tagged_document)

    print("number of documents: {}".format(len(documents)))

    model = Doc2Vec(documents, size=300, window=8, min_count=0, workers=4, alpha=0.025)

    model.train(documents, total_examples=len(documents), epochs=20)

    create_and_save_from_model(model, train_df, "features_train_doc2vec")
    create_and_save_from_model(model, test_df, "features_test_doc2vec")
    create_and_save_from_model(model, validation_df, "features_validation_doc2vec")


def create_and_save_from_model(model, df, filename):
    X = []
    Y = []
    T = []
    # values_to_write = []
    for index, row in df.iterrows():
        features_vector = model.infer_vector(row['text'])
        label = row.subreddit
        # values_to_write.append([features_vector, label])
        T.append(row.text)
        X.append(features_vector)
        Y.append(label)
    # save_obj(values_to_write, filename)
    T = np.array(T)
    X = np.array(X)
    Y = np.array(Y)
    save_obj((T, X, Y), filename)


def main():
    create_features()


if __name__ == "__main__":
    main()
