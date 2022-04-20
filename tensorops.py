#!/usr/bin/env python3

from cmath import phase
import sympy as sp
import numpy as np
from collections import OrderedDict
from itertools import product
from sympy.physics.wigner import wigner_6j as w6j
from sympy.physics.wigner import wigner_3j
from functools import lru_cache

@lru_cache(maxsize=None)
def w3j(j1, j2, j3, m1, m2, m3):
    return wigner_3j(j1, j2, j3, m1, m2, m3)

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

def CkCk(l,k):
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
