#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import logistics_pickler
from sklearn.naive_bayes import MultinomialNB


def create_predictions(train_file, test_file, output_file):
    print("loading pickled feature representation")
    T_train,X_train, Y_train = logistics_pickler.load_obj(train_file)
    T_test,X_test, Y_test = logistics_pickler.load_obj(test_file)

    print("training classifier")
    clf = MultinomialNB()
    clf.fit(X_train, Y_train)

    print("evaluating model")
    y_true = Y_test
    y_pred = clf.predict(X_test)

    logistics_pickler.save_obj((y_true, y_pred), output_file)


create_predictions("features_all_train_bow.pkl",
                   "features_all_validate_bow.pkl",
                   "predictions_bow_naive_bayes.pkl")

# create_predictions("features_train_word2vec_sum.pkl",
#                    "features_all_validate_bow.pkl",
#                    "predictions_bow_naive_bayes.pkl")
