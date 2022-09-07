#!/usr/bin/env python3

import sympy as sp

HALF = sp.S(1)/2
S_UP = HALF
S_DOWN = -HALF

element_groups = {
    'transition metals': {
                        4: list(range(21,31)),
                        5: list(range(39,49)),
                        6: list(range(71,80))
                         },
    'lanthanides': list(range(57,72)),
    'actinides': list(range(89,104))
    }

SLATER_TO_RACAH = {sp.Symbol('F^{(0)}'): sp.Symbol('A') + sp.Symbol('C')*sp.S(7)/5,
                   sp.Symbol('F^{(2)}'): 49*sp.Symbol('B') + 7*sp.Symbol('C'),
                   sp.Symbol('F^{(4)}'): sp.Symbol('C')*sp.S(63)/5}

RACAH_TO_SLATER = {sp.Symbol('A'): sp.Symbol('F^{(0)}') - sp.Symbol('F^{(4)}')/9,
                   sp.Symbol('B'): sp.Symbol('F^{(2)}')/49 - 5*sp.Symbol('F^{(4)}')/441,
                   sp.Symbol('C'): 5*sp.Symbol('F^{(4)}')/63}