#!/usr/bin/env python3

import sympy as sp
from collections import OrderedDict

spectroscopic_alphabet = 'spdfghiklmnoqrtuvwxyz'
l_from_lett_to_num = OrderedDict([(spectroscopic_alphabet[i], i) for i in range( len(spectroscopic_alphabet) )])
l_from_num_to_lett = OrderedDict([(i, spectroscopic_alphabet[i]) for i in range( len(spectroscopic_alphabet) )])

def l_notation_switch(source, upper_case = False):
    '''
    Spectroscopic notation switch.

    Parameters
    ----------
    source (str, or int): either an integer or a letter representing an angular momentum.

    Returns
    -------
    if source is (str) return is corresponding int,
    if source is (int) return is corresponding str.
    '''
    if isinstance(source, str):
        source = source.lower()
        return l_from_lett_to_num[source]
    elif isinstance(source, int):
        if upper_case:
            return l_from_num_to_lett[source].upper()
        else:
            return l_from_num_to_lett[source]

shell_alphabet = 'KLMNO'
shell_from_lett_to_num = OrderedDict([(shell_alphabet[i], i+1) for i in range( len(shell_alphabet) )])
shell_from_num_to_lett = OrderedDict([(i+1, shell_alphabet[i]) for i in range( len(shell_alphabet) )])

subshells_in_shell = {'K': ['1s'], 'L': '2s 2p'.split(' '),
                    'M': '3s 3p 3d'.split(' '), 'N': '4s 4p 4d 4f'.split(' '),
                    'O': '5s 5p 5d 5f 5g'.split(' ')}