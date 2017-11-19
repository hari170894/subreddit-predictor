#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import logistics_pickler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd

subreddits = 'r/anime r/comicbooks r/dota2 r/leagueoflegends r/conservative r/libertarian r/askscience r/explainlikeimfive r/gameofthrones r/thewalkingdead'.split()


def evaluate_predictions(predictions_filename):
    # load the pickled predictions
    y_true, y_pred = logistics_pickler.load_obj(predictions_filename)

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


evaluate_predictions("../pickle_files/predictions_bow_naive_bayes.pkl")
