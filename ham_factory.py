#!/usr/bin/env python3

from qdef import *
from misc import *
import pickle

save_to_pickle = True
pickle_fname = './data/hams.pkl'

if __name__ == '__main__':
    all_group_hams = dict()
    log = []
    for group_label in CPGs.all_group_labels:
        print(group_label)
        group_hams = {}
        for num_electrons in range(9):
            print(num_electrons)
            try:
                ham = hamiltonian_CF_CR_SO_TO(num_electrons, group_label, 2, False, True)
                group_hams[num_electrons] = ham
            except:
                log.append('Error found with group %s with %d electrons.' % (group_label, num_electrons))
        all_group_hams[group_label] = group_hams
    print(log)
    if save_to_pickle:
        print("Saving to pickle %s" % pickle_fname)
        pickle.dump(all_group_hams, open(pickle_fname, 'wb'))
