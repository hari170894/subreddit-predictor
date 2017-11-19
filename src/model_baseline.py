#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import logistics_pickler


def create_predictions(train_file, test_file, output_file):
    print("loading pickled feature representation")
    T_train, X_train, Y_train = logistics_pickler.load_obj(train_file)
    T_test, X_test, Y_test = logistics_pickler.load_obj(test_file)

    print("training classifier")
    most_common_class = max(set(Y_train), key=list(Y_train).count)

    print("evaluating model")
    y_true = Y_test
    y_pred = [most_common_class] * len(y_true)
    logistics_pickler.save_obj((T_test, y_true, y_pred), output_file)


def main():
    print("training naive bayes")

    create_predictions("features_all_train_bow.pkl",
                       "features_all_validate_bow.pkl",
                       "predictions_any_baseline.pkl")


if __name__ == "__main__":
    print("training baseline model")

    main()
