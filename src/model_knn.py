#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import logistics_pickler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def create_predictions(train_file, test_file, output_file):
    print("loading pickled feature representation")
    T_train, X_train, Y_train = logistics_pickler.load_obj(train_file)
    T_test, X_test, Y_test = logistics_pickler.load_obj(test_file)

    print("training classifier")
    param_grid = {'n_neighbors': [1]}
    clf = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=5)
    clf.fit(X_train, Y_train)
    print("Best params: {}".format(clf.best_params_))
    print("evaluating model")
    y_true = Y_test
    y_pred = clf.predict(X_test)

    logistics_pickler.save_obj((T_test, y_true, y_pred), output_file)


def main():
    print("training k nearest neighbors")

    create_predictions("features_all_train_bow.pkl",
                       "features_all_validate_bow.pkl",
                       "predictions_bow_knn.pkl")

    # create_predictions("features_train_word2vec_sum.pkl",
    #                    "features_validation_word2vec_sum.pkl",
    #                    "predictions_word2vec_sum_knn.pkl")

    # create_predictions("features_train_word2vec_max.pkl",
    #                    "features_validation_word2vec_max.pkl",
    #                    "predictions_word2vec_max_knn.pkl")

    # create_predictions("features_train_doc2vec.pkl",
    #                    "features_validation_doc2vec.pkl",
    #                    "predictions_doc2vec_knn.pkl")


if __name__ == "__main__":
    main()
