from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import pickle


def save_obj(obj, name, print_debug_info=True):
    sanitized_name = name.replace('.pkl', '')
    with open('../pickle_files/' + sanitized_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        if print_debug_info:
            print('Saved {}'.format(sanitized_name + '.pkl'))


def load_obj(name, print_debug_info=True):
    sanitized_name = name.replace('.pkl', '')
    with open('../pickle_files/' + sanitized_name + '.pkl', 'rb') as f:
        if print_debug_info:
            print('Loaded {}'.format(sanitized_name + '.pkl'))
        return pickle.load(f)
