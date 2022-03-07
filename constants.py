#!/usr/bin/env python3

import sympy as sp

S_UP = sp.S(1)/2
S_DOWN = -sp.S(1)/2
S_HALF = sp.S(1)/2

SLATER_TO_RACAH = {sp.Symbol('F^{(0)}'): sp.Symbol('A') + sp.Symbol('C')*sp.S(7)/5,
                   sp.Symbol('F^{(2)}'): 49*sp.Symbol('B') + 7*sp.Symbol('C'),
                   sp.Symbol('F^{(4)}'): sp.Symbol('C')*sp.S(63)/5}

RACAH_TO_SLATER = {sp.Symbol('A'): sp.Symbol('F^{(0)}') - sp.Symbol('F^{(4)}')/9,
                   sp.Symbol('B'): sp.Symbol('F^{(2)}')/49 - 5*sp.Symbol('F^{(4)}')/441,
                   sp.Symbol('C'): 5*sp.Symbol('F^{(4)}')/63}