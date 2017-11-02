#!/usr/bin/python3
# import os
# import sys
import time
# from collections import defaultdict
# from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# sys.path.insert(1, os.path.join(sys.path[0], '..'))


all_cols = 'linenum text id subreddit meta time author ups downs authorlinkkarma authorkarma authorisgold'.split()
use_cols = 'text subreddit'.split()

all_files = 'entertainment_anime.csv entertainment_comicbooks.csv gaming_dota2.csv gaming_leagueoflegends.csv news_conservative.csv news_libertarian.csv learning_askscience.csv learning_explainlikeimfive.csv television_gameofthrones.csv television_thewalkingdead.csv'.split()
malformed_files = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def open_with_pandas_read_csv(filename):
    start = time.time()
    if malformed_files[all_files.index(filename[7:])] == 1:
        df = pd.read_csv(filename, header=0, usecols=use_cols, names=['linenum']+all_cols, skiprows=1)
    else:
        df = pd.read_csv(filename, header=0, usecols=use_cols, names=all_cols)
    end = time.time()

    # print(filename, df[df['subreddit'] == '2'])
    # print("{} seconds elapsed".format(end - start))
    return df


def clean_data(df):
    # print("initial: {}".format(df.shape))
    df = df.dropna()
    # print("drop NaN: {}".format(df.shape))
    # df = df[df['text'] != '']
    # print("remove blanks: {}".format(df.shape))
    df = df.drop_duplicates()
    # print("drop dups: {}".format(df.shape))
    # print(df)
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
    small_sample = pd.concat([df.sample(n=100) for df in frames])
    all_data.to_csv('../res/data_sample.csv')

    print("splitting all data into train & test sets")
    train_test_splits = [train_test_split(df, test_size=0.2) for df in frames]
    training_and_validation = [train for (train, test) in train_test_splits]

    print("splitting training set into train & validation sets")
    validation_splits = [train_test_split(df, test_size=1.0/8) for df in frames]
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
