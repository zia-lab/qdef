#!/usr/bin/env python3

from itertools import product
import tensorops as to
from notation import *
from qdef import *
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

# Apr-27 2022-04-27 18:00:49

info = '''┌────────────────────────────────────────────────┐
│   This script produces the SLMSML basis for    │
│              configurations l^n.               │
│                                                │
│  The result is a dictionary whose first index  │
│  is n and whose values are dictionaries whose  │
│  keys are 3-tuples (S, L, W) corresponding to  │
│                     terms.                     │
│                                                │
│   The values of these dictionaries have keys   │
│   which are 5-tuples (W, S, MS, L, ML) that    │
│    label corresponding kets. The values of     │
│  these dictionaries are in turn 2-tuples for   │
│  which the first element is a corresponds to   │
│  2n-tuple (ms_1, ml_1, ms_2, ml_2, ... ms_n,   │
│  ml_n) and the second element an accompanying  │
│              numeric coefficient.              │
│                                                │
│  All the numbers in these expressions should   │
│       be assumed to be sympy.S numbers.        │
└────────────────────────────────────────────────┘'''

print(info)

l = 2
n_max = (2*l+1)*2 - 1
save_to_pickles = True
max_size_in_MB = 90

def ψ_maker(l, n, S_n, L_n, W_n, MS_n , ML_n, ψs_nm1, terms_nm1):
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
                    qet = qet + coeff * (Ω_bar_qet * Qet({(ml,ms):1}))
    return (Ω, qet)


if __name__ == '__main__':
    LS_Ψs = {}
    LS_Ψs[1] = {(to.HALF,l,1): {(1, to.HALF, ms, l, ml): Qet({(ml,ms):1}) for ml,ms in product(to.mrange(l), to.mrange(to.HALF)) }}

    for n in range(2, n_max+1):
        print("> Building LSMLMS determinantal qets for %s^%d" % (l_notation_switch(l).lower(),n))
        start_time = time()
        terms_n = list(to.term_range(l, n))
        terms_nm1 = list(to.term_range(l, n-1))
        all_qets = {}
        for term_n in terms_n: # for each term there will be a bunch of wavefunctions
            ψ_key = term_n
            S_n, L_n, W_n = term_n
            term_n = tuple(term_n)
            if n > 4:
                qets = Parallel(n_jobs = num_cores)(delayed(ψ_maker)(l, n, S_n, L_n, W_n, MS_n, ML_n, LS_Ψs[n-1], terms_nm1) for (MS_n, ML_n) in product(to.mrange(S_n), to.mrange(L_n)))
            else:
                qets = [ψ_maker(l, n, S_n, L_n, W_n, MS_n, ML_n, LS_Ψs[n-1], terms_nm1) for (MS_n, ML_n) in product(to.mrange(S_n), to.mrange(L_n))]
            all_qets[term_n] = dict(qets)
        LS_Ψs[n] = all_qets
        end_time = time()
        print(">> Time taken: %.2f minutes." % ((end_time-start_time)/60))
    
    if save_to_pickles:
        print("Saving to pickles...")
        LS_Ψs = {num_e: {term: {k: list(ψ.dict.items()) for k,ψ in Ψs[term].items()} for term in Ψs} for num_e, Ψs in LS_Ψs.items()}
        for n, qets in LS_Ψs.items():
            fname = './data/bases/LSMLMS_qets_%s_%d.pkl' % (l_notation_switch(l), n)
            pickle.dump(qets, open(fname,'wb'))
            pickle_size = os.path.getsize(fname)
            if pickle_size > 92_000_000:
                print("Pickling in parts.")
                split_dump(LS_Ψs, 'LSMLMS_qets_%s_%d.pkl'  % (l_notation_switch(l), n), './data/bases/', max_size_in_MB)