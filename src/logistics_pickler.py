from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import pickle


def save_obj(obj, name):
    sanitized_name = name.replace('.pkl', '')
    with open('../pickle_files/' + sanitized_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print("Saved")


def load_obj(name):
    sanitized_name = name.replace('.pkl', '')
    with open('../pickle_files/' + sanitized_name + '.pkl', 'rb') as f:
        print('Loaded')
        return pickle.load(f)
