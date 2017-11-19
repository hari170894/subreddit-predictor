#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import logistics_pickler


def main():
    print("loading pickled feature representation")
    X_train, Y_train = logistics_pickler.load_obj("../pickle_files/features_all_train_bow.pkl")
    X_test, Y_test = logistics_pickler.load_obj("../pickle_files/features_all_test_bow.pkl")

    print("training classifier")
    most_common_class = max(set(Y_train), key=list(Y_train).count)

    print("evaluating model")
    y_true = Y_test
    y_pred = [most_common_class] * len(y_true)
    logistics_pickler.save_obj((y_true, y_pred), "predictions_baseline.pkl")


if __name__ == "__main__":
    main()
