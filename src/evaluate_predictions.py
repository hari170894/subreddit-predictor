#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import logistics_pickler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


subreddits = 'r/anime r/comicbooks r/dota2 r/leagueoflegends r/conservative r/libertarian r/askscience r/explainlikeimfive r/gameofthrones r/thewalkingdead'.split()
# subreddits = 'anime comicbooks dota2 leagueoflegends conservative libertarian askscience explainlikeimfive gameofthrones thewalkingdead'.split()


def evaluate_predictions(predictions_filename):
    # load the pickled predictions
    _, y_true, y_pred = logistics_pickler.load_obj(predictions_filename)

    # put all the metrics into a table
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_pred)
    data = np.column_stack((precision, recall, fbeta_score, support))

    # compute the average of each column
    averages = np.mean([precision, recall, fbeta_score, support], axis=1)
    averages = averages.reshape((1, 4))
    data = np.concatenate((data, averages), axis=0)

    # made pandas pretty-print the table of scores
    df = pd.DataFrame(data, subreddits + ["average"], "precision recall f-score support".split())
    print(df)

    accuracy = accuracy_score(y_true, y_pred)
    print("raw accuracy: {:.6f}".format(accuracy))


def print_incorrect_predictions(predictions_filename, max_predictions_to_show):
    # load the pickled predictions
    text, y_true, y_pred = logistics_pickler.load_obj(predictions_filename)
    i = 0
    count = 0
    while i < len(y_true) and count < max_predictions_to_show:
        if y_pred[i] != y_true[i]:
            print('\nText: {}'.format(text[i]))
            print('True class      : {}'.format(y_true[i]))
            print('Predicted class : {}'.format(y_pred[i]))
            count += 1
        i += 1


def plot_confusion_matrix(predictions_filename, model_name=None):
    if model_name is None:
        model_name = predictions_filename

    text, y_true, y_pred = logistics_pickler.load_obj(predictions_filename)

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.imshow(conf_matrix, interpolation='nearest')

    plt.title("confusion matrix for {}".format(model_name))
    plt.colorbar()
    tick_marks = np.arange(len(subreddits))
    plt.xticks(tick_marks, subreddits, rotation=45)
    plt.yticks(tick_marks, subreddits)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# print("evaluating baseline")
# evaluate_predictions("predictions_any_baseline.pkl")
# print_incorrect_predictions("predictions_any_baseline.pkl", 10)
# plot_confusion_matrix("predictions_any_baseline.pkl", "Baseline")

# evaluate_predictions("predictions_bow_naive_bayes.pkl")
# print_incorrect_predictions("predictions_bow_naive_bayes.pkl", 10)
plot_confusion_matrix("predictions_bow_naive_bayes.pkl", "Naive Bayes on BOW")


# evaluate_predictions("predictions_bow_svm.pkl")
# print_incorrect_predictions("predictions_bow_svm.pkl", 10)
# plot_confusion_matrix("predictions_bow_svm.pkl", "SVM on BOW")

# evaluate_predictions("predictions_bow_linear_svm.pkl")
# print_incorrect_predictions("predictions_bow_linear_svm.pkl", 10)


# evaluate_predictions("predictions_word2vec_sum_svm.pkl")
# print_incorrect_predictions("predictions_word2vec_sum_svm.pkl", 10)

# evaluate_predictions("predictions_word2vec_max_svm.pkl")
# print_incorrect_predictions("predictions_word2vec_max_svm.pkl", 10)

# evaluate_predictions("predictions_doc2vec_svm.pkl")
# print_incorrect_predictions("predictions_doc2vec_svm.pkl", 10)

# evaluate_predictions("predictions_bow_knn.pkl")
# print_incorrect_predictions("predictions_bow_knn.pkl", 10)
# plot_confusion_matrix("predictions_bow_knn.pkl")
