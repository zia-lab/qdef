#!/usr/bin/env python3

from itertools import product
import tensorops as to
from notation import *
from qdef import *
from collections import OrderedDict
import tensorops as to
# from joblib import Parallel, delayed
# import multiprocessing
# num_cores = multiprocessing.cpu_count()

info = '''
+------------------------------------------------------------------+
|                                                                  |
|      This script produces an S,L,MS,ML basis for equivalent      |
|                   electron configurations l^n.                   |
|                                                                  |
|    After computing this basis using one-body coefficients of     |
|   fractional parentage it then uses it to calculate a S,L,J,MJ   |
|      basis by coupling the S and L of the S,L,MS,ML basis.       |
|                                                                  |
|     For the S,L,MS,ML basis the result is a dictionary whose     |
|     first index is n and whose values are dictionaries whose     |
|    keys are 3-tuples (S, L, W) corresponding to the included     |
|                              terms.                              |
|                                                                  |
|     For the S,L,J,MJ basis the result is a dictionary whose      |
|     first index is n and whose values are dictionaries whose     |
|   keys are 4-tuples (S, L, J, W) corresponding to the included   |
|                              terms.                              |
|                                                                  |
|     For the S,L,MS,ML basis the values the dictionaries have     |
|       keys which are 5-tuples (W, S, MS, L, ML) that label       |
|    corresponding kets (saved as dictionaries). The values of     |
|     these ket-dictionaries are 2-tuples for which the first      |
|     element is a 2n-tuple (ms_1, ml_1, ms_2, ml_2, ... ms_n,     |
|      ml_n) and the second element the accompanying numeric       |
|   coefficient. A second format of the results is also produced   |
|       in which the tuples are put together in Qet objects.       |
|                                                                  |
|   For the S,L,J,MJ basis the values the dictionaries have keys   |
|   which are 5-tuples (W, S, L, J, MJ) that label corresponding   |
|        kets (saved as dictionaries). The values of these         |
|   ket-dictionaries are 2-tuples for which the first element is   |
|   a 2n-tuple (ms_1, ml_1, ms_2, ml_2, ... ms_n, ml_n) and the    |
|      second element the accompanying numeric coefficient. A      |
|    second format of the results is also produced in which the    |
|             tuples are put together in Qet objects.              |
|                                                                  |
|   All the numbers in these expressions should be assumed to be   |
|                         sympy.S numbers.                         |
|                                                                  |
+------------------------------------------------------------------+'''

l = 1
n_max = 2*(2*l+1) - 1
save_to_pickles = True

def Ψ_vac(qet, elem_basis):
    '''
    This  function  removes the redundant part to a ket that is assumed to
    be  composed  of slater determinants. It does this by only picking the
    coefficients that belong to the given basis.
    '''
    new_dict = {}
    for qet_key, coeff in qet.dict.items():
        k_paired = [(qet_key[2*i], qet_key[2*i+1]) for i in range(len(qet_key)//2)]
        k_paired = tuple(k_paired)
        if k_paired in elem_basis:
            new_dict[qet_key] = coeff
    return Qet(new_dict)

def ψ_maker(l, n, S_n, L_n, W_n, MS_n , ML_n, ψs_nm1, terms_nm1, elem_basis):
    Ω = (W_n, S_n, MS_n, L_n, ML_n)
    qet = Qet({})
    for ml, ms in product(to.mrange(l), to.mrange(to.HALF)):
        Ω_1 = (1, to.HALF, ms, l, ml)
        for term_nm1 in terms_nm1:
            for Ω_bar in to.ψ_range(term_nm1):
                term_bar = (Ω_bar[1], Ω_bar[3], Ω_bar[0])
                Ω_bar_qet = ψs_nm1[term_bar][Ω_bar]
                coeff = to.Ω_coeff(l,n, Ω, Ω_bar, Ω_1)
                if coeff:
                    delta_qet = (Ω_bar_qet * Qet({(ml,ms):1}))
                    qet = qet + coeff * delta_qet
    good_qet = Ψ_vac(qet, elem_basis)
    good_qet = sp.sqrt(n)*good_qet
    return (Ω, good_qet)

def SMSLML_gen(l, n_max, verbose=False):
    '''
    This function creates the coupled kets {S, MS, L, ML}
    in terms of .... 
    '''
    assert n_max <= 2*(2*l+1), "n_max don't make no sense ..."
    SMSLMS_Ψs = OrderedDict()
    SMSLMS_Ψs[1] = OrderedDict({(to.HALF,l,1): {(1, to.HALF, ms, l, ml): Qet({(ml,ms):1}) 
                            for ml,ms in product(to.mrange(l), to.mrange(to.HALF)) }})

    for num_electrons in range(2, n_max+1):
        if verbose:
            print("> Building LSMLMS determinantal qets for %s^%d" 
                  % (l_notation_switch(l).lower(),num_electrons))
        elem_basis = elementary_basis("multi equiv electron", l, num_electrons)
        start_time = time()
        terms_n = list(to.term_range(l, num_electrons))
        terms_nm1 = list(to.term_range(l, num_electrons-1))
        all_qets = OrderedDict()
        for term_n in terms_n: # for each term there will be a bunch of wavefunctions
            S_n, L_n, W_n = term_n
            term_n = tuple(term_n)
            qets = [ψ_maker(l, num_electrons, S_n, L_n, W_n, MS_n, ML_n, 
                            SMSLMS_Ψs[num_electrons-1], terms_nm1, elem_basis) 
                    for (MS_n, ML_n) in product(to.mrange(S_n), to.mrange(L_n))]
            all_qets[term_n] = dict(qets)
        SMSLMS_Ψs[num_electrons] = all_qets
        end_time = time()
        if verbose:
            print(">> Time taken: %.2f minutes." % ((end_time-start_time)/60))
    return SMSLMS_Ψs

if __name__ == '__main__':
    print(info)
    SMSLMS_Ψs = SMSLML_gen(l, n_max, True)
    LSJMJ_Ψs = dict([(num_electrons, to.SMSLML_to_SLJMJ(SMSLMS_Ψs[num_electrons])) for num_electrons in SMSLMS_Ψs])
    if save_to_pickles:
        print("> Saving to pickles...")
        for n, qets in SMSLMS_Ψs.items():
            fname = './data/bases/L-S-ML-MS_detqets_%s_%d.pkl' % (l_notation_switch(l), n)
            print(">> Saving as qets to %s..." % fname)
            pickle.dump(qets, open(fname,'wb'))
        for n, qets in LSJMJ_Ψs.items():
            fname = './data/bases/L-S-J-MJ_detqets_%s_%d.pkl' % (l_notation_switch(l), n)
            print(">> Saving as qets to %s..." % fname)
            pickle.dump(qets, open(fname,'wb'))
        SMSLMS_Ψs = {num_e: {term: {k: list(ψ.dict.items()) for k,ψ in Ψs[term].items()} for term in Ψs} for num_e, Ψs in SMSLMS_Ψs.items()}
        LSJMJ_Ψs = {num_e: {term: {k: list(ψ.dict.items()) for k,ψ in Ψs[term].items()} for term in Ψs} for num_e, Ψs in LSJMJ_Ψs.items()}
        for n, qets in SMSLMS_Ψs.items():
            fname = './data/bases/L-S-ML-MS_detdicts_%s_%d.pkl' % (l_notation_switch(l), n)
            print(">> Saving as dictionaries to %s..." % fname)
            pickle.dump(qets, open(fname,'wb'))
        for n, qets in LSJMJ_Ψs.items():
            fname = './data/bases/L-S-J-MJ_detdicts_%s_%d.pkl' % (l_notation_switch(l), n)
            print(">> Saving as dictionaries to %s..." % fname)
            pickle.dump(qets, open(fname,'wb'))
