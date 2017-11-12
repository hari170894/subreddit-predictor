#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 3

all_cols = 'linenum text id subreddit meta time author ups downs authorlinkkarma authorkarma authorisgold'.split()
use_cols = 'text subreddit'.split()

all_files = 'entertainment_anime.csv entertainment_comicbooks.csv gaming_dota2.csv gaming_leagueoflegends.csv news_conservative.csv news_libertarian.csv learning_askscience.csv learning_explainlikeimfive.csv television_gameofthrones.csv television_thewalkingdead.csv'.split()
malformed_files = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def open_with_pandas_read_csv(filename):
    if malformed_files[all_files.index(filename[7:])] == 1:
        df = pd.read_csv(filename, header=0, usecols=use_cols, names=['linenum'] + all_cols, skiprows=1)
    else:
        df = pd.read_csv(filename, header=0, usecols=use_cols, names=all_cols)
    return df


def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def file_to_dataframe(filename):
    df = open_with_pandas_read_csv(filename)
    df = clean_data(df)
    return df


def main():
    print("Reading all files")
    frames = [file_to_dataframe('../res/' + filename) for filename in all_files]
    all_data = pd.concat(frames)
    all_data.to_csv('../res/data_all.csv')

    print("Creating small sample for testing purposes")
    small_sample = pd.concat([df.sample(n=100, random_state=RANDOM_STATE) for df in frames])
    small_sample.to_csv('../res/data_sample.csv')

    print("splitting all data into train & test sets")
    train_test_splits = [train_test_split(df, test_size=0.2, random_state=RANDOM_STATE) for df in frames]
    training_and_validation = [train for (train, test) in train_test_splits]

    print("splitting training set into train & validation sets")
    validation_splits = [train_test_split(df, test_size=1.0 / 8, random_state=RANDOM_STATE) for df in training_and_validation]
    training = [train for (train, valid) in validation_splits]
    validation = [valid for (train, valid) in validation_splits]

    training = pd.concat(training)
    testing = pd.concat([test for (train, test) in train_test_splits])
    validation = pd.concat(validation)

    training.to_csv('../res/data_train_all.csv')
    testing.to_csv('../res/data_test_all.csv')
    validation.to_csv('../res/data_validate_all.csv')


if __name__ == "__main__":
    main()
