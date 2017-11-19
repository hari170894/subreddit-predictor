#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals


def load_data():
    import logistics_download_data
    logistics_download_data.main()

    import logistics_prepare_data
    logistics_prepare_data.main()


def load_features():
    import features_bow
    features_bow.main()
    print("-" * 30)

    import features_doc2vec
    features_doc2vec.main()
    print("-" * 30)

    import features_word2vec
    features_word2vec.main()
    print("-" * 30)


def load_models():
    import model_baseline
    model_baseline.main()
    print("-" * 30)

    # import model_conv_nn
    # model_conv_nn.main()
    print("-" * 30)

    # import model_fully_connected_nn
    # model_fully_connected_nn.main()
    print("-" * 30)

    # import model_knn
    # model_knn.main()
    print("-" * 30)

    import model_naive_bayes
    model_naive_bayes.main()
    print("-" * 30)

    # import model_random_forest
    # model_random_forest.main()
    print("-" * 30)

    import model_svm
    model_svm.main()
    print("-" * 30)



print("Task selector (quickly run portion of machine learning pipeline)")
print("=" * 30)
print("0 - run everything (download data, extract features, train models)")
print("1 - download and split data into train, test, validation")
print("2 - run all feature extractors on data")
print("3 - train all models on all feature representations (not all models finished)")

user_choice = raw_input()
if user_choice == "0":
    load_data()
    load_features()
    load_models()
elif user_choice == "1":
    load_data()
elif user_choice == "2":
    load_features()
elif user_choice == "3":
    load_models()
