#!/usr/bin/env python3

import sympy as sp
from collections import OrderedDict
from itertools import product
from sympy.physics.wigner import wigner_6j as w6j
from sympy.physics.wigner import wigner_3j
from functools import lru_cache
import os
import pickle
from notation import *
from qdef import mrange, lrange
from qdefcore import Qet

module_dir = os.path.dirname(__file__)

seniority_dict = pickle.load(open(os.path.join(module_dir,'data','seniority_dict.pkl'),'rb'))
all_terms = pickle.load(open(os.path.join(module_dir,'data','all_term_labels.pkl'),'rb'))

HALF = sp.S(1)/2

@lru_cache(maxsize=None)
def w3j(j1, j2, j3, m1, m2, m3):
    return wigner_3j(j1, j2, j3, m1, m2, m3)

def tp1(x):
    return 2*x+1

@lru_cache(maxsize=None)
def VC_coeff(bra,ket):
    '''
    REF: Judd 1-22
    '''
    j1, m1, j2, m2 = bra
    j1, j2, j3, m3 = ket
    phase = phaser(j1,-j2,m3)
    return sp.sqrt(2*j3+1) * phase * w3j(j1,j2,j3,m1,m2,-m3)

def phaser(*parts):
    '''
    returns (-1)**(sum(parts))
    '''
    total_exponent = sum(parts)
    if total_exponent % 2 == 0:
        return 1
    else:
        return -1

def paired_kron(*pairs):
    '''
    A quick pairwise kronecker delta
    '''
    boo = [pairs[idx] != pairs[idx+1] for idx in range(0,len(pairs),2)]
    if any(boo):
        return 0
    else:
        return 1

def Ck(k, l, lp):
    '''
    Returns the reduced matrix element
     (l||Cᴷ||l)
    '''
    phase = phaser(l)
    threejay = w3j(l, k, lp, 0, 0, 0)
    return phase * threejay * sp.sqrt((2*l+1) * (2*lp+1))

def CkCk(l, k):
    '''
    Returns the matrix elements of the operator
    (γ j1 j2 J MJ | Cᴷ⋅Cᴷ | γp j1p j2p Jp MJp)
    from which the Coulomb repulsion operator may
    be constructed for two electrons.

    Parameters
    ----------
    l (int)
    k (int)

    Returns
    -------
    rmat, rdict (tuple)
        rmat (sp.Matrix)
        rdict (OrderedDict)
    '''
    Ss = [0,1]
    Ls = range(0,2*l+1)
    ckckdict = OrderedDict()
    for S, L, S̃, L̃ in product(Ss, Ls, Ss, Ls):
        MSs = range(-S, S+1)
        MLs = range(-L, L+1)
        if (S+L) % 2 !=0:
            continue
        if (S̃+L̃) % 2 !=0:
            continue
        for MS, MS̃, ML, ML̃ in product(MSs, MSs, MLs, MLs):
            key = (S,MS,L,ML,S̃,MS̃,L̃,ML̃)
            ckckdict[key] = 0
            if paired_kron(S, S̃, L, L̃, MS, MS̃, ML, ML̃):
                phase = phaser(l,l,L)
                sixjay = w6j(l,l,k,l,l,L)
                reduced_Ck = Ck(k, l, l)
                ckckdict[key] = phase * sixjay * reduced_Ck**2
    return ckckdict

def H1(l):
    H1dict = OrderedDict()
    for k in [0,2,4,6]:
        ckck = CkCk(l,k)
        for key, val in ckck.items():
            if key not in H1dict:
                H1dict[key] = 0
            else:
                H1dict[key] += sp.Symbol('F^{(%d)}' % k) * val
    return H1dict
    
def sl(l):
    Ss = [0,1]
    Ls = range(0, 2*l+1)
    sldict = OrderedDict()
    for S, L, S̃, L̃ in product(Ss, Ls, Ss, Ls):
        Js = range(abs(L-S),L+S+1)
        J̃s = range(abs(L̃-S̃),L̃+S̃+1)
        if (S+L) % 2 !=0 or (S̃+L̃) % 2 !=0:
            continue
        for J, J̃ in product(Js, J̃s):
            MJs = range(-J, J+1)
            MJ̃s = range(-J̃, J̃+1)
            for MJ, MJ̃ in product(MJs, MJ̃s):
                key = (S,L,J,MJ,S̃,L̃,J̃,MJ̃)
                sldict[key] = 0
                if paired_kron(J, J̃, MJ, MJ̃):
                    phase = phaser(S̃, L, J)
                    sixjay = w6j(S,S̃,1, L̃,L,J)
                    red_s = reduced_s(S, S̃)
                    red_l = reduced_l(l, L, L̃)
                    sldict[key] = 2*sp.Symbol('\zeta') * phase * sixjay * red_s * red_l
    return sldict

def reduced_s(S, Sp):
    '''Returns the values of the reduced
    matrix element. This is specialized for the
    spin of the electron.
    (S||Ŝ||Sp)'''
    half = sp.S(1)/2
    phase = phaser(1, Sp, 1)
    rooty = sp.sqrt(2*S+1) * sp.sqrt(2*Sp+1)
    sixj = w6j(S, 1, Sp, half, half, half)
    radius = sp.sqrt(sp.S(3)/2)
    return phase * sixj * rooty * radius

def reduced_l(l, L, Lp):
    '''Returns the values of the reduced
    matrix element
    (L||l̂||Lp)'''
    phase = phaser(l, l, Lp, 1)
    rooty = sp.sqrt(2*L+1) * sp.sqrt(2*Lp+1)
    sixj = w6j(L, 1, Lp, l, l, l)
    radius = sp.sqrt(l*(l+1)*(2*l+1))
    return phase * sixj * rooty * radius

def reduced_j(l, L, Lp):
    '''Returns the values of the reduced
    matrix element of an angular momentum j
    (L||l̂||Lp)'''
    phase = phaser(l, l, Lp, 1)
    rooty = sp.sqrt(2*L+1) * sp.sqrt(2*Lp+1)
    sixj = w6j(L, 1, Lp, l, l, l)
    radius = sp.sqrt(l*(l+1)*(2*l+1))
    return phase * sixj * rooty * radius

def term_range(l,n):
    '''
    Given an electron configuration l^n this
    iterator provides the corresponding terms.
    '''
    if n > 2*l+1:
        n = (4*l+2) - n
    terms = all_terms[(all_terms['l'] == l) & (all_terms['n'] == n)][['S','L','idx']].to_records(index=False)
    return terms

def ψ_range(term):
    '''
    Given a term this yields the wavefunction
    contained in it
    '''
    S, L, W = term
    ML_s, MS_s = mrange(L), mrange(S)
    for MS, ML in product(MS_s, ML_s):
        Ω  = (W,  S,  MS,  L,  ML) # full address of row state
        yield Ω

def config_range(l,n):
    '''
    '''
    for term in term_range(l,n):
        for Ω in ψ_range(term):
            yield Ω

@lru_cache(maxsize=None)
def Ω_coeff(l, n, Ω, Ωp_1, Ωp_2):
    '''
    This function returns the coefficient
    <lⁿ⁻¹ W* S* MS* L* ML*; l ms ml | lⁿW S MS L ML>
    Parameters
    ----------
    n (int): number of electrons
    Returns
    -------
    '''
    (W,    S,    MS,    L,    ML)    = Ω # daughter
    (W_p,  S_p,  MS_p,  L_p,  ML_p)  = Ωp_1 # parent1
    (W_pp, S_pp, MS_pp, L_pp, ML_pp) = Ωp_2 # parent2
    cfp_args = (l, n,
               S,    L,    W, # daughter term
               S_p,  L_p,  W_p, # parent1 term
               S_pp, L_pp, W_pp) # parent2 term
    # Vector Coupling coeffs
    vcS = VC_coeff((S_p, MS_p, S_pp, MS_pp), 
                      (S_p, S_pp, S,    MS))
    if vcS == 0:
        return 0
    
    vcL = VC_coeff((L_p, ML_p,  L_pp, ML_pp), 
                      (L_p, L_pp, L,    ML))
    if vcL == 0:
        return 0
    
    cfp = CFP_1(*cfp_args, True)
    if cfp == 0:
        return 0
    val = vcS * vcL * cfp
    return val

def CFP_fun(num_bodies, string_notation = False):
    '''
    Parameters
    ----------
    num_bodies (int): either 1,2,3,4
    string_notation (bool): if true then input to function is a friendly string

    Returns
    -------
    fun (function): a function that provides the coefficients of
    fractional parentage for the provided number of bodies.

    References
    ----------
    + Data is from Velkov, “Multi-Electron Coefficients of Fractional Parentage for the p, d, and f Shells.”
    '''
    if num_bodies not in [1,2,3,4]:
        raise Exception("%d bodies not included in minimal set." % num_bodies)
    if not string_notation:
        doc_string = '''
        Returns the {num_bodies}-body coefficient of fractional parentage.  If
        coefficient  is  physical  but  not immediately available, a series of
        identities are tried out to see if it can be computed thus.

        Parameters
        ----------
        l (int): orbital angular momentum of constituent electrons
        n (int): how many electrons in configuration
        S (half-int or int): S of daughter
        L (int): L of daughter
        W (int): index of daughter
        S_p (half-int or int): S of parent_1
        L_p (int): L of parent_1
        W_p (int): index of parent_1
        S_pp (half-int or int): S of parent_2
        L_pp (int): L of parent_2
        W_pp (int): index of parent_2
        fill_missing (bool): if True then 0 is returned for key not found, unsafe.
        
        Returns
        -------
        CFP (sp.S): symbolic expression for coefficent of fractional parentage
        '''.format(num_bodies = num_bodies)
        def fun(l, n, 
                S,    L,    W, # daughter
                S_p,  L_p,  W_p, # parent_1
                S_pp, L_pp, W_pp, # parent_2
                fill_missing=False):
            n_p  = n - num_bodies
            n_pp = num_bodies
            key  = (num_bodies, l, 
                   n,    S,    L,    W,
                   n_p,  S_p,  L_p,  W_p,
                   n_pp, S_pp, L_pp, W_pp)
            if key in fun.data: # Case I
                return fun.data[key]
            else:  # Case II
                if n <= 2*l+1:
                    ν    = seniority_dict[(l, n,    S,    L,    W)]
                else:
                    ν    = seniority_dict[(l, 4*l+2-n,    S,    L,    W)]
                if n_p <= 2*l+1:
                    ν_p  = seniority_dict[(l, n_p,  S_p,  L_p,  W_p)]
                else:
                    ν_p  = seniority_dict[(l, 4*l+2-n_p,  S_p,  L_p,  W_p)]
                if n_pp <= 2*l+1:
                    ν_pp = seniority_dict[(l, n_pp, S_pp, L_pp, W_pp)]
                else:
                    ν_pp = seniority_dict[(l, 4*l+2-n_pp, S_pp, L_pp, W_pp)]
                alter_key = (num_bodies, l, 
                            4*l + 2 - n_p, S_p, L_p, W_p,
                            n_pp, S_pp, L_pp, W_pp, 
                            4*l + 2 - n, S, L, W)
                if alter_key in fun.data:
                    phase = phaser(sp.S(ν - ν_p - (n % 2) + (n_p % 2))/2)
                    phase = (phase
                            * sp.sqrt(tp1(S_p)*tp1(L_p)/tp1(S)/tp1(L))
                            * sp.sqrt(sp.binomial(4*l + 2 - n_p, n_pp)/sp.binomial(n, n_pp))
                            )
                    return phase * fun.data[alter_key]
                else: # Case III
                    alter_key = (num_bodies, l, 
                                4*l + 2 - n_pp, S_pp, L_pp, W_pp,
                                4*l + 2 - n, S, L, W, 
                                n_p, S_p, L_p, W_p)
                    if alter_key in fun.data:
                        phase = phaser(sp.S(ν - ν_pp - (n % 2) + (n_pp % 2))/2)
                        phase = (phase
                                * sp.sqrt(tp1(S_pp) * tp1(L_pp) / tp1(S) / tp1(L))
                                * sp.sqrt(sp.binomial(4*l + 2 - n_pp, n_p) / sp.binomial(n, n_p))
                                )
                        return phase * fun.data[alter_key]
                    else: # Case IV
                        alter_key = (num_bodies, l, 
                                    4*l + 2 - n_pp, S_pp, L_pp, W_pp,
                                    n_p, S_p, L_p, W_p, 
                                    4*l + 2 - n, S, L, W)
                        if alter_key in fun.data:
                            phase = phaser(S + S_p + S_pp + L + L_p + L_pp + sp.S(ν + ν_pp + (n_p%2))/2)
                            phase = (phase
                                    * sp.sqrt(tp1(S_pp) * tp1(L_pp) / tp1(S) / tp1(L))
                                    * sp.sqrt(sp.binomial(4*l + 2 - n_pp, n_p) / sp.binomial(n, n_p))
                                    )
                            return phase * fun.data[alter_key]
                        else: # Case V
                            alter_key = (num_bodies, l, 
                                        4*l + 2 - n_p, S_p, L_p, W_p,
                                        4*l + 2 - n, S, L, W, 
                                        n_pp, S_pp, L_pp, W_pp)
                            if alter_key in fun.data:
                                phase = phaser(S, + S_p - S_pp + L + L_p - L_pp, + sp.S(ν + ν_p - (n_pp%2))/2)
                                phase = (phase
                                        * sp.sqrt(tp1(S_p) * tp1(L_p) / tp1(S) / tp1(L))
                                        * sp.sqrt(sp.binomial(4*l + 2 - n_p, n_pp) / sp.binomial(n, n_pp))
                                        )
                                return phase * fun.data[alter_key]
                            else: # Case VI
                                alter_key = (num_bodies, l, 
                                            n, S, L, W,
                                            n_pp, S_pp, L_pp, W_pp, 
                                            n_p, S_p, L_p, W_p)
                                if alter_key in fun.data:
                                    phase = phaser(n_p*n_pp + S_p + S_pp -S + L_p + L_pp -L)
                                    return phase * fun.data[alter_key]
                                else:
                                    if fill_missing:
                                        return 0
                                    else:
                                        raise Exception("Missing key : %s" % str(key))
        fun.__doc__ = doc_string
        print("Loading data for %d-body coefficients of fractional parentage..." % (num_bodies))
        fun.data = pickle.load(open(os.path.join(module_dir, 'data', 'CFP_%s-body-dict.pkl' % num_bodies),'rb'))
    else:
        doc_string = '''
        Returns the {num_bodies}-body coefficient of fractional parentage.  If
        coefficient  is  physical  but  not immediately available, a series of
        identities are tried out to see if it can be computed thus.

        Parameters
        ----------
        string_input  (str):  in  format  ('l n daughter_term parent_term_1
        parent_term_2'). For example 'd 5 2I1 1I1 2D1'
        fill_missing (bool): if True then 0 is returned for key not found, unsafe.

        NOTE: it is assumed that terms with W_max = 1 still include the index. For example,
        if 2D only has a single term, it should be input as 2D1.

        Returns
        -------
        CFP (sp.S): symbolic expression for coefficent of fractional parentage.

        '''.format(num_bodies = num_bodies)
        def fun(string_input, fill_missing=False):
            l, n, daughter_term, parent_term_1, parent_term_2 = string_input.split(' ')
            l, n = l_notation_switch(l), int(n)
            S = sp.S(int(daughter_term[0]) - 1)/2
            L = l_notation_switch(daughter_term[1])
            W = int(daughter_term[2])
            S_p = sp.S(int(parent_term_1[0]) - 1)/2
            L_p = l_notation_switch(parent_term_1[1])
            W_p = int(parent_term_1[2])

            S_pp = sp.S(int(parent_term_2[0]) - 1)/2
            L_pp = l_notation_switch(parent_term_2[1])
            W_pp = int(parent_term_2[2])

            n_p = n - num_bodies
            n_pp = num_bodies
            key = (num_bodies, l, 
                   n, S, L, W,
                   n_p, S_p, L_p, W_p,
                   n_pp, S_pp, L_pp, W_pp)
            if key in fun.data: # Case I
                return fun.data[key]
            else:  # Case II
                ν    = seniority_dict[(l, n,    S,    L,    W)]
                ν_p  = seniority_dict[(l, n_p,  S_p,  L_p,  W_p)]
                ν_pp = seniority_dict[(l, n_pp, S_pp, L_pp, W_pp)]
                alter_key = (num_bodies, l, 
                            4*l + 2 - n_p, S_p, L_p, W_p,
                            n_pp, S_pp, L_pp, W_pp, 
                            4*l + 2 - n, S, L, W)
                if alter_key in fun.data:
                    phase = phaser(sp.S(ν - ν_p - (n % 2) + (n_p % 2))/2)
                    phase = (phase
                            * sp.sqrt(tp1(S_p)*tp1(L_p)/tp1(S)/tp1(L))
                            * sp.sqrt(sp.binomial(4*l + 2 - n_p, n_pp)/sp.binomial(n, n_pp))
                            )
                    return phase * fun.data[alter_key]
                else: # Case III
                    alter_key = (num_bodies, l, 
                                4*l + 2 - n_pp, S_pp, L_pp, W_pp,
                                4*l + 2 - n, S, L, W, 
                                n_p, S_p, L_p, W_p)
                    if alter_key in fun.data:
                        phase = phaser(sp.S(ν - ν_pp - (n % 2) + (n_pp % 2))/2)
                        phase = (phase
                                * sp.sqrt(tp1(S_pp)*tp1(L_pp)/tp1(S)/tp1(L))
                                * sp.sqrt(sp.binomial(4*l + 2 - n_pp, n_p) / sp.binomial(n, n_p))
                                )
                        return phase * fun.data[alter_key]
                    else: # Case IV
                        alter_key = (num_bodies, l, 
                                    4*l + 2 - n_pp, S_pp, L_pp, W_pp,
                                    n_p, S_p, L_p, W_p, 
                                    4*l + 2 - n, S, L, W)
                        if alter_key in fun.data:
                            phase = phaser(S + S_p + S_pp + L + L_p + L_pp + sp.S(ν + ν_pp + (n_p%2))/2)
                            phase = (phase
                                    * sp.sqrt(tp1(S_pp)*tp1(L_pp)/tp1(S)/tp1(L))
                                    * sp.sqrt(sp.binomial(4*l + 2 - n_pp, n_p) / sp.binomial(n, n_p))
                                    )
                            return phase * fun.data[alter_key]
                        else: # Case V
                            print('V')
                            alter_key = (num_bodies, l, 
                                4*l + 2 - n_p, S_p, L_p, W_p,
                                4*l + 2 - n, S, L, W, 
                                n_pp, S_pp, L_pp, W_pp)
                            if alter_key in fun.data:
                                phase = phaser(S, + S_p - S_pp + L + L_p - L_pp, + sp.S(ν + ν_p - (n_pp%2))/2)
                                phase = (phase
                                        * sp.sqrt(tp1(S_p)*tp1(L_p)/tp1(S)/tp1(L))
                                        * sp.sqrt(sp.binomial(4*l + 2 - n_p, n_pp) / sp.binomial(n, n_pp))
                                        )
                                return phase * fun.data[alter_key]
                            else: # Case VI
                                alter_key = (num_bodies, l, 
                                            n, S, L, W,
                                            n_pp, S_pp, L_pp, W_pp, 
                                            n_p, S_p, L_p, W_p)
                                if alter_key in fun.data:
                                    phase = phaser(n_p*n_pp + S_p + S_pp -S + L_p + L_pp -L)
                                    return phase * fun.data[alter_key]
                                else:
                                    if fill_missing:
                                        return 0
                                    else:
                                        raise Exception("Missing key : %s" % str(key))
        fun.__doc__ = doc_string
        print("Loading data for %d-body coefficients of fractional parentage..." % (num_bodies))
        fun.data = pickle.load(open(os.path.join(module_dir, 'data', 'CFP_%s-body-dict.pkl' % num_bodies),'rb'))
    return fun

# first added on Apr-28 2022-04-28 17:52:05
def SMSLML_to_SLJMJ(LS_basis):
    '''
    This  function  receives  a  dictionary  for  a  basis  in |S,MS,L,ML⟩
    coupling and returns the corresponding basis in |S,L,J,MJ⟩ coupling.
    This  is  done  by  bringing  in  the vector coupling coefficients for
    coupling the L and S angular momenta of a given term.
    When  this  is carried through all of the the SL terms, there might be
    repetitions  of  a  given  (SLJ)  term,  these are discriminated by an
    additional index W that distinguishes them.
    Parameters
    ----------
    LS_basis  (dict): keys are 3-tuples (S, L, W), values are dictionaries
    whose  keys  are  5-tuples (W, S, MS, L, ML) and whose values are qets
    whose  keys  are 2n-tuples of ms,ml pairs (ms_1, ml_1, ms_2, ml_2, ...
    ms_n, ml_n).
    Returns
    -------
    LSJ_basis  (dict):  keys  are  3-tuples  (S,L,J,WJ),  whose values are
    dictionaries  whose  keys  are 5-tuples (S,L,J,MJ,WJ) and whose values
    are qets whose keys are 2n-tuples ms,ml pairs (ms_1, ml_1, ms_2, ml_2,
    ... ms_n, ml_n).
    '''
    SLJM_terms = {}
     # this is just to keep in check the SLJ terms that appear and 
     # distinguish between the different ones that may appear.
    term_index = {}
    for term, term_qets in LS_basis.items():
        S, L, W = term
        for J in lrange(S,L):
            SLJM_sub_term = (S,L,J)
            if SLJM_sub_term not in term_index:
                term_index[SLJM_sub_term] = 1
            WJ = term_index[SLJM_sub_term]
            SLJM_term = (S,L,J,WJ)
            term_index[SLJM_sub_term] += 1
            SLJM_terms[SLJM_term] = {}
            MJs = mrange(J)
            for MJ in MJs:
                sum_qet = Qet({})
                sum_qet_key = (S,L,J,MJ,WJ)
                for qet_key, qet in term_qets.items():
                    (W, S, MS, L, ML) = qet_key
                    SLvc = VC_coeff((S, MS, L, ML), 
                                        (S, L, J, MJ))
                    sum_qet += SLvc*qet
                SLJM_terms[SLJM_term][sum_qet_key] = sum_qet
    return SLJM_terms


CFP_1 = CFP_fun(1)
