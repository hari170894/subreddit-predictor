#!/usr/bin/python3
from tqdm import tqdm
import requests


# Code used from https://stackoverflow.com/a/37573701/2230446
def download(url, filename):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))

    with open(filename, 'wb') as f:
        for data in tqdm(r.iter_content(32 * 1024), total=total_size, unit='B', unit_scale=True):
            f.write(data)


def main():
    all_files = 'entertainment_anime.csv entertainment_comicbooks.csv gaming_dota2.csv gaming_leagueoflegends.csv news_conservative.csv news_libertarian.csv learning_askscience.csv learning_explainlikeimfive.csv television_gameofthrones.csv television_thewalkingdead.csv'.split()

    for filename in all_files:
        url = 'https://raw.githubusercontent.com/linanqiu/reddit-dataset/master/' + filename
        full_filename = '../res/' + filename
        print('Downloading {}'.format(filename))
        download(url, full_filename)


if __name__ == "__main__":
    main()
