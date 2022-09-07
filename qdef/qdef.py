#!/usr/bin/env python

######################################################################
#                                   __     ____                      #
#                        ____ _____/ /__  / __/                      #
#                       / __ `/ __  / _ \/ /_                        #
#                      / /_/ / /_/ /  __/ __/                        #
#                      \__, /\__,_/\___/_/                           #
#                        /_/                                         #
#                                                                    #
######################################################################


from xmlrpc.client import MAXINT
import sympy as sp
import numpy as np
import pandas as pd
from sympy.physics.quantum import Bra, Ket, KetBase, TensorProduct
from itertools import combinations, permutations, combinations_with_replacement
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter
from qdef.constants import *
from qdef.notation import *
from qdef.qdefcore import *
import pickle, os
from sympy import Eijk as εijk
plot_style = 'default'
plt.style.use(plot_style)
from functools import lru_cache


module_dir = os.path.dirname(__file__)

name_to_symb = pickle.load(open(os.path.join(module_dir,'data',
                                            'name_to_symb.pkl'),'rb'))
name_to_num  = pickle.load(open(os.path.join(module_dir,'data',
                                            'name_to_num.pkl'),'rb'))
symb_to_name  = pickle.load(open(os.path.join(module_dir,'data',
                                            'symb_to_name.pkl'),'rb'))
symb_to_num  = pickle.load(open(os.path.join(module_dir,'data',
                                            'symb_to_num.pkl'),'rb'))
num_to_name  = pickle.load(open(os.path.join(module_dir,'data',
                                            'num_to_name.pkl'),'rb'))
num_to_symb  = pickle.load(open(os.path.join(module_dir,'data',
                                            'num_to_symb.pkl'),'rb'))

atomicGoodies  = pickle.load(open(os.path.join(module_dir,'data',
                                            'atomicGoodies.pkl'),'rb'))
ionization_data  = pickle.load(open(os.path.join(module_dir,'data',
                                            'ionization_data.pkl'),'rb'))['data']
ionic_radii_df = pd.read_csv(os.path.join(module_dir,'data','ionic_radii.csv'),comment='#')


atom_symbs   = list(symb_to_name.keys())
atom_names   = list(name_to_num.keys())
s2_operators = pickle.load(open(os.path.join(module_dir,'data','s2_operators.pkl'),'rb'))

nistdf_levels = pd.read_pickle(os.path.join(module_dir,'data',
                                    'nist_atomic_spectra_database_levels.pkl'))
nistdf_lines = pd.read_pickle(os.path.join(module_dir,'data',
                                    'nist_atomic_spectra_database_lines.pkl'))

spinData = pd.read_pickle(os.path.join(module_dir,'data','spindata.pkl'))

crystal_splits_loc = os.path.join(module_dir,'data','crystal_splits3p8.pkl')
crystal_splits = pickle.load(open(crystal_splits_loc,'rb'))

# =============================================================== #
# ===================== Load group theory data ================== #

group_dict = pickle.load(open(os.path.join(module_dir,'data','gtpackdata.pkl'),'rb'))
double_group_dict = pickle.load(open(os.path.join(module_dir,'data','gtpackdata_double_groups.pkl'),'rb'))

group_data = group_dict['group_data']
double_group_data = double_group_dict['group_data']
group_metadata = group_dict['metadata']
double_group_metadata = double_group_dict['metadata']

# ===================== Load group theory data ================== #
# =============================================================== #

regen_fname = os.path.join(module_dir,'data','CPGs.pkl')
crystal_fields_fname = os.path.join(module_dir,'data','CPGs.pkl')

if os.path.exists(regen_fname):
    print("Reloading %s ..." % regen_fname)
    CPGs = pickle.load(open(os.path.join(module_dir,'data','CPGs.pkl'),'rb'))
else:
    print("%s not found, regenerating ..." % regen_fname)
    CPGs = CPGroups(group_data, double_group_data)
    pickle.dump(CPGs, open(os.path.join(module_dir,'data','CPGs.pkl'),'wb'))

# ===================== Load group theory data ================== #
# =============================================================== #

def phaser(*summands):
    '''
    Returns (-1)**(sum(parts)).
    It  assumes  that  the  sum is an integer, and doesn't check
    that this is true.

    Parameters
    ----------
    *summads: any number of numbers that must add up to an integer.

    Returns
    -------
    (-1)**(sum(summands))
    '''
    total_exponent = sum(summands)
    if total_exponent % 2 == 0:
        return 1
    else:
        return -1

def tp1(x):
    '''
    two-plus-one
    Returns 2*x + 1 assuming that x is integer of half-integer.
    '''
    return 2*x + 1

def kron(*args):
    '''
    Returns 1 if all args are equal, 0 otherwise.

    Parameters
    ----------
    *args (poly): any number of things that may be compared.

    Returns
    -------
    1 if all args are equal, 0 otherwise.
    '''
    num_args = len(args)
    if num_args == 2:
        return 1 if args[0]==args[1] else 0
    elif num_args == 3:
        return 1 if (args[0]==args[1]==args[2]) else 0
    elif num_args == 4:
        return 1 if (args[0]==args[1]==args[2]==args[3]) else 0
    else:
        return 1 if all([arg == args[0] for arg in args]) else 0

@lru_cache(maxsize=None)
def Wigner_d(l, n, m, β):
    '''
    Wigner_d

    Reference
    ---------
    - Bradley and Cracknell 2.1.6
    '''
    k_min = max(0, m-n)
    k_max = min(l-n, l+m)
    Wig_d_prefact = sp.sqrt((sp.factorial(l + n)
                           * sp.factorial(l + m)
                           * sp.factorial(l - n)
                           * sp.factorial(l - m)))
    Wig_d_summands = [((-sp.S(1))**(k - m + n)
                       * sp.cos(β/2)**(2*l + m - n - 2*k)
                       * sp.sin(β/2)**(2*k + n - m)
                       / sp.factorial(l - n - k)
                       / sp.factorial(l + m - k)
                       / sp.factorial(k)
                       / sp.factorial(k - m + n)
                      )
                      for k in range(k_min, k_max+1)]
    Wig_d = (Wig_d_prefact * sum(Wig_d_summands)).doit()
    return Wig_d

@lru_cache(maxsize=None)
def Wigner_D(l, n, m, alpha, beta, gamma):
    '''
    Wigner_D matrix element with Condon-Shortley phase.

    This  is  for  Euler angles referenced against fixed axes in
    the  z-y-z  convention.  In this convention there is first a
    rotation  about  the  z-axis by alpha, then a rotation about
    the  y-axis by beta, and finally a rotation about the z-axis
    by gamma. All these axes being fixed all throughout.

    This function assumes that the spherical harmonics carry the
    (-1)ᵐ Condon-Shortley phase.

    R(α, β, γ) Yₗᵐ = Σₙ Wigner_D(l, n, m, α, β, γ) Yₗⁿ
    with n = (-l, -l+1, ..., l-1, l)
    
    Parameters
    ----------
    l (int): ≥ 0 
    n (int): ∈ {-l, -l+1, ... , l-1, l}
    m (int): ∈ {-l, -l+1, ... , l-1, l}
    alpha (float): with 0 ≤ α ≤ 2π
    beta  (float): with 0 ≤ β ≤ π
    gamma (float): with 0 ≤ γ ≤ 2π

    Returns
    -------
    Wid_D (sp.S or float)

    Reference
    ---------
    Bradley  and  Cracknell  2.1.4  (with  added Condon-Shortley
    phase)
    '''
    if beta == 0:
      if n == m:
        Wig_D = (sp.cos(m*alpha + m*gamma) - sp.I * sp.sin(m*alpha + m*gamma))
      else:
        Wig_D = 0
    elif beta == sp.pi:
      if n == -m:
        if l % 2 == 0:
          Wig_D =   (sp.cos(-m*alpha + m*gamma) + sp.I * sp.sin(-m*alpha + m*gamma))
        else:
          Wig_D = - (sp.cos(-m*alpha + m*gamma) + sp.I * sp.sin(-m*alpha + m*gamma))
      else:
        Wig_D = 0
    else:
      Wig_D_0 = sp.I**(abs(n) + n - abs(m) - m) # this always evaluates to a real number
      Wig_D_1 = (sp.cos(-n*gamma - m*alpha) \
                 + sp.I * sp.sin(-n * gamma - m * alpha)) * Wigner_d(l,n,m,beta)
      Wig_D = Wig_D_0 * Wig_D_1
    # This is where the Condon-Shortley phase is added.
    if (n + m) % 2 == 1:
        Wig_D = - Wig_D
    return Wig_D


def mrange(j):
    '''
    Range of m values for given j.

    Give a j get a list with corresponding mj.

    Parameters
    ----------
    j  (int or half-int): angular momentum

    Returns
    -------
    (tuple) [-j, -j+1, ..., j-1, j]
    '''
    # returns ranges that work for half integers
    assert int(j*2) == 2*j, "j should be integer or half-integer"
    assert j>=0, "j should be non-negative"
    j = sp.S(int(2*j))/2
    return tuple(-j+i for i in range(2*j+1))

def lrange(j1,j2):
    '''
    Total j values when adding j1 and j2 angular momenta.

    When adding j1, and j2, return range of possible total
    angular momenta.
    
    Parameters
    ----------
    j1:(int or half-int) ≥ 0
    j2:(int or half-int) ≥ 0

    Returns
    -------
    (tuple) (|j1-j2|,|j1-j2|+1, ..., j1+j2)
    '''
    assert j1>=0, "j1 should be non-negative"
    assert j2>=0, "j1 should be non-negative"
    assert int(j1*2) == 2*j1, "j1 should be integer or half-integer"
    assert int(j2*2) == 2*j2, "j1 should be integer or half-integer"
    j1 = sp.S(int(2*j1))/2
    j2 = sp.S(int(2*j2))/2
    return tuple(abs(j1-j2) + i for i in range((j1+j2)-abs(j1-j2)+1))

def elementary_basis(kind: str, l_orbital: int, num_electrons = None, 
                              flatten = False, reverse_order = False,
                              as_qets = False) -> list:
    '''
    Basis of different kinds for multi-electron systems.

    These bases are elementary in the sense that they are not built within
    any  coupling scheme. They only take into account if the electrons are
    equivalent or not.

    (single electron)
    A single electron basis is simply all the combinations of the possible
    ml,  and  ms  for the given orbital angular momentum l. This basis has
    (2*l+1)*2 elements.

    (multi inequiv electron)
    A  multi  inequivalent  electron basis is composed of all the possible
    combinations of num_electrons of the single electron basis. Here it is
    assumed  that  the  inequivalent electrons still have the same orbital
    angular momentum l. This basis has ((2l+1)*2)^num_electrons elements.

    (multi equiv electron)
    A   multi-equiv-electron   basis  is  composed  of  all  the  possible
    combinations  of  num_electrons  of the single electron basis, with no
    repeated  (ml,  ms)  single  electron  states,  and  with the ordering
    considered  to  be  irrelevant.  This  basis has sp.binomial(2*(2l+1),
    num_electrons) elements.

    (multi equiv electron ordered)
    A   multi-equiv-electron   basis  is  composed  of  all  the  possible
    combinations  of  num_electrons  of the single electron basis, with no
    repeated  (ml,  ms)  single electron states. Different from the multi-
    euquiv-electron  basis the ordering here is considered to be relevant.
    This  basis  has sp.binomial(2*(2l+1), num_electrons) * num_electrons!
    elements.

    Parameters
    -----------
    kind  (str):  ∈ ["single electron", "multi inequiv electron",
      "multi equiv electron", or "multi equiv electron ordered"].

    l_orbital  (int): orbital angular momentum number.

    num_electrons (int): how many electrons are in the mix.

    flatten   (bool):   if  True  the  returned  tuples  for  the  ml,  ms
    combinations  are  flattened  into  single tuples. That is, instead of
    returning
      [((-1, -1/2), (0, 1/2)), ((-1, -1/2), (2, 1/2)), ...] 
    it instead returns
      [(-1, -1/2, 0, 1/2)), (-1,-1/2,2,1/2), ...] 
    
    reverse_order (bool): if True the the ms goes before the ml.

    Returns
    -------
    A list with tuples representing allowed combinations of 
    ((ml_1, ms_1), (ml_2, ms_2), ... (ml_n, ms_n)) 
    '''
    assert isinstance(l_orbital,int) or isinstance(l_orbital, sp.core.numbers.Integer)
    if kind == "single electron":
        if reverse_order:
            return [(ms, ml) for ms in [S_DOWN, S_UP] for ml in range(-l_orbital,l_orbital+1)]
        else:
            return [(ml, ms) for ml in range(-l_orbital,l_orbital+1) for ms in [S_DOWN, S_UP]]
    elif kind == "multi inequiv electron":
        assert num_electrons != None, "num_electrons missing"
        single_e_basis = elementary_basis("single electron", l_orbital, None, flatten, reverse_order)
        return_basis = list(product(*[single_e_basis for _ in range(num_electrons)]))
    elif kind == "multi equiv electron":
        assert num_electrons != None, "num_electrons missing"
        single_e_basis = elementary_basis("single electron", l_orbital, None, flatten, reverse_order)
        return_basis = list(combinations(single_e_basis, num_electrons))
    elif kind == "multi equiv electron ordered":
        assert num_electrons != None, "num_electrons missing"
        single_e_basis = elementary_basis("single electron", l_orbital, None, flatten, reverse_order)
        return_basis = list(permutations(single_e_basis, num_electrons))
    if flatten:
        return_basis = [sum(vec, tuple()) for vec in return_basis]
    if as_qets:
        return_basis = [Qet({qnums:1}) for qnums in return_basis]
    return return_basis

def threeHarmonicIntegral(l1, m1, l2, m2, l3, m3):
    '''
    ∫ Yl1m1* Yl2m2 Yl3m3 dΩ

    Returns the value of the three spherical harmonics integral,
    the variety with the first one having a complex conjugate.

    This  function  assumes that the spherical harmonics include
    the Condon-Shortley phase.

    It is non-zero if:
    - l1 + l2 + l2 is even,
    - |l1-l2| <= l3 <= l1+l2.

    .. math:: \int d\Omega Y_{l_1,m_1}^* Y_{l_2,m_2} Y_{l_3,m_3}

    Parameters
    ----------
    l1 (int)
    m1 (int) ∈ (-l1, -l1+1, ... , l1-1, l1)
    l2 (int)
    m2 (int) ∈ (-l2, -l2+1, ... , l2-1, l2)
    l3 (int)
    m3 (int) ∈ (-l3, -l3+1, ... , l3-1, l3)
    
    Returns
    -------
    (sp.S)

    '''
    from sympy.physics.wigner import gaunt
    return sp.S(-1)**m1 * gaunt(l1,l2,l3,-m1,m2,m3)

# angular momentum

def LS_allowed_terms(l:int, n:int) -> dict:
    '''
    LS allowed terms

    Calculate   the   allowed   terms   in  LS  coupling  for  homogeneous
    configurations of equivalent electrons.

    Parameters
    ----------
    l (int): orbital angular momentum
    n (int): how many electrons

    Returns
    -------
    terms (dict) with keys equal to (2S+1) multiplicities and values equal
    to  Counter  objects  of  allowed  total  angular  momenta  (in string
    notation).
    '''
    def flatten(nlist):
        flist = []
        for elem in nlist:
            for it in elem:
                flist.append(it)
        return flist
    ls = [l]*n
    spins = [-1/2, 1/2]
    terminators = {0:'S',1:'P',2:'D',3:'F',4:'G',5:'H',6:'I',7:'K',8:'L',
                   9:'M',10:'N',11:'O',12:'Q',13:'R',14:'T',15:'U',16:'V',
                  17:'W',18:'X',19:'Y',20:'Z'}
    single_states = []
    mLs = list(range(-l,l+1))
    for mL in mLs:
        for mS in [-1/2,1/2]:
            single_states.append((mL,mS))
    configs = list(map(set,list(combinations(single_states,n))))
    MLs = range(-sum(ls),sum(ls)+1)
    spins = np.arange(-1/2*len(ls),1/2*len(ls)+1)
    microstates = {}
    for ML in MLs:
        subconfigs = [config for config in configs if sum([l[0] for l in list(config)]) == ML]
        for mtot in spins:
            thestates = [list(config)[:len(ls)*2] for config in subconfigs if sum([l[1] for l in list(config)])==mtot]
            if len(thestates) > 0:
                microstates[(ML,mtot)] = list(map(flatten,thestates))
            else:
                microstates[(ML,mtot)] = []
    # find the non-empty ones
    # from those pick the coordinates that are closest to the lower left corner
    # if it is possible to to diagonally to the upper right, then this is a boxy box
    # if not, then it is a rowy row
    # it might also be a columny col
    collections = []
    while True:
        non_empty = [[k,abs(MLs[0]-k[0])+abs(spins[0]-k[1])] for k in microstates.keys() if len(microstates[k])>0]
        if len(non_empty) == 0:
            break
        corner = non_empty[np.argsort([r[-1] for r in non_empty])[0]][0]
        if corner == (0,0):
            case = 'box'
            start = (0,0)
            end = (0,0)
        else:
            right = (corner[0]+1, corner[1])
            up = (corner[0], corner[1]+1)
            diag = (corner[0]+1, corner[1]+1)
            if up in microstates.keys():
                up_bool = len(microstates[up]) > 0
            else:
                up_bool = False
            if right in microstates.keys():
                right_bool = len(microstates[right]) > 0
            else:
                right_bool = False
            if diag in microstates.keys():
                diag_bool = len(microstates[diag]) > 0
            else:
                diag_bool = False
            if diag_bool and up_bool and right_bool:
                case = 'box'
                start = corner
                end = (-corner[0], -corner[1])
            elif up_bool and not right_bool:
                case = 'col'
                start = corner
                end = (corner[0],-corner[1])
            else:
                case = 'row'
                start = corner
                end = (-corner[0], corner[1])
        if case == 'row':
            collect = []
            for k in np.arange(start[0], end[0]+1):
                collect.append(microstates[(k,0)].pop())
        elif case == 'col':
            collect = []
            for k in np.arange(start[1], end[1]+1):
                collect.append(microstates[(start[0],k)].pop())
        elif case == 'box':
            collect = []
            for k in np.arange(start[0], end[0]+1):
                for l in np.arange(start[1],end[1]+1):
                    collect.append((microstates[(k,l)].pop()))
        collections.append(collect)
    terms = {}
    for collection in collections:
        L = max(np.sum(np.array(collection)[:,::2],axis=1))
        S = max(np.sum(np.array(collection)[:,1::2],axis=1))
        if int(S) == S:
            S = int(S)
        multiplicity = int(2*S+1)
        if multiplicity in terms.keys():
            terms[multiplicity].append(terminators[L])
        else:
            terms[multiplicity] = [terminators[L]]
    return {k: Counter(v) for k,v in terms.items()}

###########################################################################
####################### Angular Momentum Matrices #########################

@lru_cache(maxsize=None)
def Jmatrices(j, high_to_low = False, as_dict = False):
    '''
    Angular momentum matrices for the given value of j. Using ħ=1.

    By  default  the  matrix  elements  (i.e.  the  top-left corner of the
    matrix)  corresponds to the most negative value of the projection onto
    the z-axis.

    Parameters
    ----------
    j (int or half-int): angular momentum
    high_to_low  (Bool): if True mj = j is on top left corner of matrices
    as_dict (Bool): whether to return a dictionary instead
    
    Returns
    -------
    if as_dict == False:
        (Jx, Jy, Jz) (tuple) whose elements are sp.Matrix
    else:
        (Jxdict, Jydict, Jzdict) (tuple) whose elements are dictionaries
        whose keys are tuples (mjrow, mjcol) and whose values are the 
        corresponding matrix elements.

    '''
    j = sp.S(int(2*j))/2
    basis = mrange(j)
    if high_to_low:
        basis = basis[-1::-1]
    lp = sp.Matrix([(Qet({mj+1: sp.sqrt((j-mj)*(j+mj+1))})).vec_in_basis(basis) for mj in basis]).T
    lm = sp.Matrix([(Qet({mj-1: sp.sqrt((j+mj)*(j-mj+1))})).vec_in_basis(basis) for mj in basis]).T
    lx = (lp + lm)/sp.S(2)
    ly = (lp - lm)/(sp.S(2)*sp.I)
    lz = sp.Matrix([(Qet({mj: mj})).vec_in_basis(basis) for mj in basis]).T
    if not as_dict:
        return (lx, ly, lz)
    else:
        lxyz_dicts = tuple({(ml0, ml1): op[basis.index(ml0), basis.index(ml1)] for ml0 in basis for ml1 in basis}
                        for op in (lx, ly, lz))
        return lxyz_dicts


@lru_cache(maxsize=None)
def LSmatrix(l, s, high_to_low = False, as_dict = False):
    '''
    L⋅S for a single electron

    Provide the matrix representation for L⋅S = Lx*Sx + Ly*Sy + Lz*Sz
    in the product basis 
    [(ml1, ms1), (ml1, ms2), ... , (ml1, msn),
     (ml2, ms1), (ml2, ms2), ... , (ml2, msn),
     ...
     (mlm, ms1), (ml2, ms2), ... , (ml2, msn),
     ]
    
    Uses ħ=1.

    Parameters
    ----------
    l (int or half-int): orbital angular momentum
    s (int or half-int): spin angular momentum
    high_to_low (Bool): if True then ml = l is on top left corner of matrices
    as_dict (Bool): if True then the function returns a dictionary of matrix elements
    
    Returns
    -------
    if as_dict == False
        Lsmat (sp.Matrix) a matrix for the L⋅S operator
    else:
        Lsdict where the keys are two tuples ((ml_row, ms_row), (ml_col, ms_col)) and
        whose values are the corresponding matrix elements.

    '''
    Lmatrices = Jmatrices(l, high_to_low)
    Smatrices = Jmatrices(s, high_to_low)
    LSparts = [TensorProduct(Lm, Sm) for Lm, Sm in zip(Lmatrices, Smatrices)]
    Lsmat = sum(LSparts, sp.zeros(int(2*l+1) * int(2*s+1)))
    if not as_dict:
        return Lsmat
    else:
        if high_to_low:
            lsbasis = list(product(mrange(l)[-1::-1], mrange(s)[-1::-1]))
        else:
            lsbasis = list(product(mrange(l), mrange(s)))
        Lsdict = {((mlrow, msrow),(mlcol, mscol)): Lsmat[lsbasis.index((mlrow, msrow)), lsbasis.index((mlcol, mscol))] 
                 for (mlrow, msrow) in lsbasis for (mlcol, mscol) in lsbasis}
        return Lsdict

@lru_cache(maxsize=None)
def Ltotal(l, num_electrons):
    '''
    This function returns the matrix representation of the orbital angular
    momentum  operator  for  the  given  number  of  electrons,  using the
    uncoupled equivalent electron basis.

    Parameters
    ----------
    l (int): orbital angular momentum
    num_electrons (int): how many equivalent electrons

    Returns
    -------
    (Lx, Ly, Lz) (tuple)
        Lx (sp.Matrix): matrix representation of Lx in the uncoupled basis
        Ly (sp.Matrix): matrix representation of Ly in the uncoupled basis
        Lz (sp.Matrix): matrix representation of Lz in the uncoupled basis
    '''
    base = elementary_basis('multi equiv electron', l, num_electrons, as_qets=True)
    bbase = elementary_basis('multi equiv electron', l, num_electrons)

    Jx, Jy, Jz = Jmatrices(l)
    Jx = Jx.T.tolist()
    Jy = Jy.T.tolist()
    Jz = Jz.T.tolist()
    Jxdict = {ml: dict(zip(mrange(l), Jx[idx])) for idx, ml in enumerate(mrange(l))}
    Jydict = {ml: dict(zip(mrange(l), Jy[idx])) for idx, ml in enumerate(mrange(l))}
    Jzdict = {ml: dict(zip(mrange(l), Jz[idx])) for idx, ml in enumerate(mrange(l))}

    Lxparts = [opSingleMulti(Jxdict, idx) for idx in range(num_electrons)]
    Lyparts = [opSingleMulti(Jydict, idx) for idx in range(num_electrons)]
    Lzparts = [opSingleMulti(Jzdict, idx) for idx in range(num_electrons)]
    Lxlist, Lylist, Lzlist = [], [], []
    for idx, baseqet in enumerate(base):
        nqet = sum([baseqet.apply(lx) for lx in Lxparts], Qet())
        vqet = nqet.vec_in_detbasis(bbase)
        Lxlist.append(vqet)
        nqet = sum([baseqet.apply(ly) for ly in Lyparts], Qet())
        vqet = nqet.vec_in_detbasis(bbase)
        Lylist.append(vqet)
        nqet = sum([baseqet.apply(lz) for lz in Lzparts], Qet())
        vqet = nqet.vec_in_detbasis(bbase)
        Lzlist.append(vqet)
    return (sp.Matrix(Lxlist).T, sp.Matrix(Lylist).T, sp.Matrix(Lzlist).T)

@lru_cache(maxsize=None)
def L_squared(l, num_electrons):
    '''
    Returns  the  matrix  representaiton  of  Lx^2  +  Ly^2  + Lz^2 in the
    uncoupled basis.

    This  matrix  is  diagonalized by the LSMLMS and JMJSMS bases. It is a
    square matrix of dimension sp.binomial(2*(2l+1), num_electrons)
    
    Parameters
    ----------
    l (int): orbital angular momentum
    num_electrons (int): how many equivalent electrons

    Returns
    -------
    L^2 (sp.Matrix): Lx**2 + Ly**2 + Lz**2
    '''
    Lx, Ly, Lz = Ltotal(l, num_electrons)
    return (Lx**2 + Ly**2 + Lz**2)

####################### Angular Momentum Matrices #########################
###########################################################################

def single_electron_braket(qet0, qet1):
    '''
    Given  two qets qet0 and qet1 composed of determinantal states, and an
    assumed single particle operator op that is extended by addition to an
    n-particle operator, this function returns the value of the braket
    
      <qet0| \sum_1^N op_i |qet1>

    in terms of brakets of single electron orbitals.

    Spin is assumed to be integrated in the notation for the symbols where
    a  symbol  that  is  adorned with an upper bar is assumed to have spin
    down and one without to have spin up.

    Parameters
    ----------
    qet0       (qdefcore.Qet): a qet
    qet1       (qdefcore.Qet): another qet

    Returns
    -------
    full_braket  (qdefcore.Qet): with each key having three symbols, first
    one  equal  to  a  single  electron  orbital,  second one equal to the
    provided  single electron operator, and the third one equal to another
    single electron orbital. Interpreted as <φi | (op)* | φj>.

    *  The  operator  is omitted and is assumed to be in the middle of the
    two symbols.

    References
    ----------
    -   "Multiplets of Transition-Metal Ions in Crystals", Chapter 3
        Sugano, Tanabe, and Kamimura
    '''
    full_braket = []
    qet0 = qet0.dual()
    for det0, coeff0 in qet0.dict.items():
        set0 = set(det0)
        for det1, coeff1 in qet1.dict.items():
            # before  given  value to the braket it is necessary
            # to  align  the symbols in the determinantal states
            # and keep track of the reordering sign
            set1 = set(det1)
            # there should be no repeat symbols in any determinantal state
            assert len(set0) == len(det0) and len(set1) == len(det1), "There's something funny here ..."
            common_symbs = list(set0.intersection(set1))
            different_symbs0 = [x for x in det0 if x not in common_symbs]
            different_symbs1 = [x for x in det1 if x not in common_symbs]
            newdet0 = different_symbs0 + common_symbs
            newdet1 = different_symbs1 + common_symbs
            ordering0 = [det0.index(x) for x in newdet0]
            ordering1 = [det1.index(x) for x in newdet1]
            extra_sign = εijk(*ordering0) * εijk(*ordering1)
            total_coeff = coeff0 * coeff1 * extra_sign
            odet0 = newdet0
            odet1 = newdet1
            comparisons = list(map(lambda x: x[0]==x[1], zip(odet0, odet1)))
            if all(comparisons):
                # CASE I
                single_brakets = [((φ, φ), total_coeff) for φ in odet0]
            elif (odet0[0] != odet1[0]) and all(comparisons[1:]):
                # CASE II
                single_brakets = [((odet0[0],
                                        odet1[0]), total_coeff)]
            elif (odet0[0] != odet1[0]) and (odet0[1] != odet1[1]):
                # CASE III
                single_brakets = []
            else:
                raise Exception("Ooops, This case shouldn't occur")
            full_braket.extend(single_brakets)
    full_braket = sum([Qet({k:v}) for k,v in full_braket],Qet({}))
    return full_braket

###########################################################################
###################### Magnetic Dipole Calculations #######################

def mag_dip_xyz(axis, l):
    '''
    Parameters
    ----------
    axis (str) ∈ {'x', 'y', 'z'}
    l    (int) : orbital angular momentum

    Returns
    -------
    mag_dip_fun (function) a function which can be used to evaluate matrix
    elements  in  qets  which  represent a matrix element for the magnetic
    dipole  operator  taking  the  standard  basis  as the single electron
    basis used to build multi-electron determinantal states.
    '''
    axis_index = {'x':0, 'y':1, 'z':2}[axis]
    L_orbital = Jmatrices(l, False, as_dict=True)
    L_spin = Jmatrices(1/2, False, as_dict=True)
    def mag_dip_fun(qnums, coeff):
        orb0, orb1 = qnums
        gs = sp.Symbol('g_s', real=True)
        orbital_contrib, spin_contrib = 0, 0
        if orb0.spin == orb1.spin:
            orbital_contrib = L_orbital[axis_index][(orb0.orbital[1],
                                                    orb1.orbital[1])]
        if orb0.orbital == orb1.orbital:
            spin_contrib = L_spin[axis_index][(orb0.spin, orb1.spin)]
        return {1: coeff*(orbital_contrib + gs * spin_contrib)}
    return mag_dip_fun

def standard_mag_dip(num_electrons, l):
    '''
    Magnetic dipole operator matrix elements.

    Compute  the  matrix  elements  of the magnetic dipole operator in the
    standard basis for a system with the given number of electrons.

        μ = -\mu_B * (L + g_s S)

    Parameters
    ----------
    num_electrons (int) :
    l             (l)   :

    Returns
    -------
    mag_dip_matrices, mag_dip_operators (tuple) with
        mag_dip_matrices  (OrderedDict): with keys 'x', 'y', 'z' and
        values  sp.Matrix 
        mag_dip_operators (OrderedDict): with keys
        'x',  'y',  'z' and values OrderedDict whose keys are tuples
        with as many SpinOrbital as num_electrons there are.
    
    '''
    if (num_electrons,l) in standard_mag_dip.values:
        return standard_mag_dip.values[(num_electrons, l)]
    mag_dip = {axis: mag_dip_xyz(axis, l) for axis in 'xyz'}
    
    single_e_spin_orbitals = [SpinOrbital((l,m), spin) 
                                for spin in [S_DOWN, S_UP] for m in mrange(l)]
    slater_dets = list(combinations(single_e_spin_orbitals, num_electrons))
    slater_qets = [Qet({k:1}) for k in slater_dets]

    mag_dip_operator = OrderedDict()
    mag_dip_operator['x'] = OrderedDict()
    mag_dip_operator['y'] = OrderedDict()
    mag_dip_operator['z'] = OrderedDict()

    mag_dip_matrices = OrderedDict([('x',[]), ('y',[]), ('z',[])])
    for qet0 in slater_qets:
        row = {'x':[], 'y':[], 'z':[]}
        qet0key = list(qet0.dict.keys())[0]
        for qet1 in slater_qets:
            qet1key = list(qet1.dict.keys())[0]
            key = (qet0key, qet1key)
            for axis in mag_dip:
                if (qet1key, qet0key) in mag_dip_operator[axis]:
                    braket = sp.conjugate(mag_dip_operator[axis][(qet1key, qet0key)])
                    mag_dip_operator[axis][key] = braket
                    row[axis].append(braket)
                else:
                    braket = -single_electron_braket(qet0, qet1).apply(mag_dip[axis]).as_symbol_sum()
                    mag_dip_operator[axis][key] = braket
                    row[axis].append(braket)
        for axis in row:
            mag_dip_matrices[axis].append(row[axis])
    mag_dip_matrices = {axis: sp.Matrix(mag_dip_matrices[axis]) for axis in mag_dip_matrices}
    standard_mag_dip.values[(num_electrons, l)] = (mag_dip_matrices, mag_dip_operator)
    return mag_dip_matrices, mag_dip_operator
standard_mag_dip.values = {}

###################### Magnetic Dipole Calculations #######################
###########################################################################

def double_braket_basis_change(braket, basis_changer):
    '''
    Take  a  qet,  understood  as  a  four  symbol braket, and a
    dictionary  that  maps  the  current basis to a new one, and
    return  the resulting expression for the new braket in terms
    of  the new basis. All throughout it is assumed that between
    the given braket there is an implicit operator.

    Parameters
    ----------
    braket   (qdefcore.Qet)
    basis_changer (dict):  keys being  equal  to single electron
    quantum  symbols  and  values  to  qets  that  determine the
    superposition of the new basis to which this vector is being
    mapped to. The keys of the dictionary need not  include  all
    the quantum symbols included in the qet.

    Returns
    -------
    new_braket (qdefcore.Qet)

    Example
    -------

    braket = Qet({(1,2,3,4): 5,
                (8,4,3,1): 1})
    basis_change = {1: Qet({(8,): sp.I})}
    print(double_braket_basis_change(braket, basis_change))
    >> {(8, 2, 3, 4): -5*I, (8, 4, 3, 8): I}

    '''

    new_braket = Qet({})
    for k, v in braket.dict.items():
        βi, βj, βk, βl = [(Qet({(γ,):1}) if γ not in basis_changer else basis_changer[γ]) for γ in k]
        βi, βj = βi.dual(), βj.dual()
        γiγjγkγl = ((βi * βj) * βk) * βl
        new_braket = new_braket + ( v * γiγjγkγl)
    return new_braket

def to_slater_params(qnums, coeff):
    '''
    This function will take a set of qnums that are assumed to be l1, m1, l2, m2, l1p, m1p, l2p, m2p
    and will return a set of qnums that correspond to Slater integrals and corresponding coefficients.
    '''
    l1, m1, s1, l2, m2, s2, l1p, m1p, s1p, l2p, m2p, s2p = qnums
    if not(kron(s1,s1p) and kron(s2,s2p)):
        return {}
    elif kron(m1+m2, m1p+m2p):
        if (m1 - m1p) % 2 == 0:
            phase = 1
        else:
            phase = -1
        new_dict = {}
        for k in range(6):
            key = sp.Symbol('F^{(%d)}' % k)
            c1 = (sp.sqrt((4*sp.pi) / (2*k+1)) 
                    * threeHarmonicIntegral(l1,   m1,
                                            k,   (m1 - m1p),
                                            l1p, m1p))
            c2 = (sp.sqrt((4*sp.pi) / (2*k+1)) 
                    * threeHarmonicIntegral(l2,  m2,
                                            k,   (m2-m2p),
                                            l2p, m2p))
            val = phase * coeff * c1 * c2
            if val:
                new_dict[key] = val
        return new_dict
    else:
        return {}

def double_electron_braket(qet0, qet1):
    '''
    Given  two  qets,  which  are  assumed to be composed of determinantal
    states, and a two electron operator op, return value of the braket

      <qet0| \sum_{i>j=1}^N f_i,j |qet1> 
    
    in terms of brakets of double electron orbitals.

    Parameters
    ----------
    qet0    (qdefcore.Qet): a qet of determinantal states
    qet1    (qdefcore.Qet): a qet of determinantal states

    Returns
    -------
    full_braket  (qdefcore.Qet):  with each key having five symbols, first
    two equal to a two electron orbitals, middle one equal to the provided
    double  electron  operator,  and the last two equal to another pair of
    two  single  electron  orbitals;  interpreted as <φi, φj | (op)* | φk,
    φl>. 
    
    *  The  operator  is omitted and is assumed to be in the middle of the
    four symbols.

    References
    ----------
    -   "Multiplets of Transition-Metal Ions in Crystals", Chapter 3
        Sugano, Tanabe, and Kamimura
    '''

    full_braket = []
    qet0 = qet0.dual()
    for det0, coeff0 in qet0.dict.items():
        num_electrons = len(det0)
        set0 = set(det0)
        for det1, coeff1 in qet1.dict.items():
            # before giving a value to the braket it is necessary to align the symbols 
            # in the determinantal states and keep track of the reordering sign
            set1 = set(det1)
            common_symbs = list(set0.intersection(set1))
            different_symbs0 = [x for x in det0 if x not in common_symbs]
            different_symbs1 = [x for x in det1 if x not in common_symbs]
            # there are no repeat symbols in any determinantal state
            newdet0 = different_symbs0 + common_symbs
            newdet1 = different_symbs1 + common_symbs
            ordering0 = [det0.index(x) for x in newdet0]
            ordering1 = [det1.index(x) for x in newdet1]
            extra_sign = εijk(*ordering0) * εijk(*ordering1)
            total_coeff = extra_sign * coeff0 * coeff1
            odet0, odet1 = newdet0, newdet1
            double_brakets = []
            comparisons = list(map(lambda x: x[0]==x[1], zip(odet0, odet1)))
            if all(comparisons):
                # CASE I
                for i in range(num_electrons):
                    for j in range(i+1,num_electrons):
                        double_brakets.append(((odet0[i], odet0[j],
                                                odet0[i], odet0[j]),
                                                total_coeff))
                        double_brakets.append(((odet0[i], odet0[j],
                                                    odet0[j], odet0[i]),
                                                    -total_coeff))
            elif (odet0[0] != odet1[0]) and all(comparisons[1:]):
                # CASE II
                for j in range(1,num_electrons):
                    double_brakets.append(((odet0[0], odet0[j],
                                            odet1[0], odet0[j]),
                                            total_coeff))
                    double_brakets.append(((odet0[0], odet0[j],
                                            odet0[j], odet1[0]),
                                            -total_coeff))
            elif (odet0[0] != odet1[0]) and (odet0[1] != odet1[1]) and all(comparisons[2:]): 
                # CASE III
                double_brakets.append(((odet0[0], odet0[1],
                                            odet1[0], odet1[1]),
                                            total_coeff))
                double_brakets.append(((odet0[0], odet0[1],
                                            odet1[1], odet1[0]),
                                            -total_coeff))
            elif not any(comparisons[:3]):
                # CASE IV
                # print("IV")
                double_brakets = []
            else:
                raise Exception("Ooops, This case shouldn't occur")
            full_braket.extend(double_brakets)
    full_braket = sum([Qet({k:v}) for k,v in full_braket],Qet({}))
    return full_braket

###########################################################################
############################## Hamiltonians ###############################

def S_squared(num_electrons:int, l:int, sparse: bool = False):
    '''
    Provides  the  matrix  representation of the S^2 operator in
    the determinantal uncoupled basis.

    Parameters
    ----------
    num_electrons  : how many electrons.
    l              : orbital angular momentum of electrons.
    sparse         : whether to return a sparse sympy Matrix.

    Returns
    -------
    s2operator (sp.Matrix)

    '''
    if l == 2:
        return s2_operators[num_electrons]
    ssquared_dictionaire = trees_dict(HALF)
    # Using uncoupled spherical harmonics basis.
    single_e_basis = [SpinOrbital(sp.Symbol('Y_{%d,%d}' % (l,m)), spin) 
                        for spin in [S_DOWN, S_UP] for m in range(-l,l+1)]
    basis_change = OrderedDict([(SpinOrbital(sp.Symbol('Y_{%d,%d}' % (l,m)), spin), Qet({(l,m,spin):1})) 
                        for spin in [S_DOWN, S_UP] for m in range(-l,l+1)])

    def ssquared_op_two_bod(qnums, coeff):
        l1, m1, s1, l2, m2, s2, l1p, m1p, s1p, l2p, m2p, s2p = qnums
        if not(kron(m1,m1p) and kron(m2,m2p)):
            return {}
        else:
            if (s1,s2,s1p,s2p) in ssquared_dictionaire:
                return {1: coeff * ssquared_dictionaire[(s1,s2,s1p,s2p)]}
            else:
                return {}

    def ssquared_op_one_bod(qnums, coeff):
        the_dict = {1:0}
        γ1, γ2 = qnums
        m1 = list(basis_change[γ1].dict.keys())[0][1]
        m2 = list(basis_change[γ2].dict.keys())[0][1]
        s1, s2  = γ1.spin, γ2.spin
        if m1 == m2 and s1 == s2:
            the_dict[1] = coeff * HALF * (HALF + 1)
        if the_dict[1] == 0:
            return {}
        else:
            return the_dict

    # add spin up and spin down
    single_e_spin_orbitals = single_e_basis
    # create determinantal states
    slater_dets = list(combinations(single_e_spin_orbitals, num_electrons))
    slater_qets = [Qet({k:1}) for k in slater_dets]

    if sparse:
        s2operator = {}
    else:
        s2operator = []

    for idx0, qet0 in enumerate(slater_qets):
        row = []
        for idx1, qet1 in enumerate(slater_qets):
            double_braket = double_electron_braket(qet0, qet1)
            double_braket = double_braket_basis_change(double_braket, basis_change)
            ssquared_matrix_element_twobody = sp.expand(double_braket.apply(ssquared_op_two_bod).as_symbol_sum())
            single_braket = single_electron_braket(qet0, qet1)
            ssquared_matrix_element_onebod = single_braket.apply(ssquared_op_one_bod).as_symbol_sum()
            matrix_element = (ssquared_matrix_element_twobody + ssquared_matrix_element_onebod)
            if matrix_element != 0:
                if sparse:
                    s2operator[(idx0,idx1)] = (matrix_element)
                else:
                    row.append(matrix_element)
            else:
                if not sparse:
                    row.append(matrix_element)
        if not sparse:
            s2operator.append(row)

    if sparse:
        s2operator = (sp.SparseMatrix(len(slater_qets), len(slater_qets), s2operator))
    else:
        s2operator = sp.Matrix(s2operator)
    return s2operator

def trees_dict(l, return_matrix = False):
    '''
    This  function  returns the matrix elements of the l_1⋅l_2 operator in
    the standard basis for two electrons. Here l_1 is the angular momentum
    operator for one electron,  l_2  the angular   momentum operator   for
    another electron, both of them having the same l.

    This  can  be  used to evaluate the two-body contribution of the Trees
    effective operator.

    The  keys are given in the format (m1, m2, m1p, m2p) so that the given
    value    in    the    dictionary    equals    the    matrix    element
    <m1,m2|l_1⋅l_2|m1p,m2p>.

    Parameters
    ----------
    l (int) : angular momentum of the two electrons

    Returns
    -------
    trees_dictionare (dict): keys in the  format  (m1, m2, m1p, m2p)  and
    values equal to the corresponding matrix element 
        <m1,m2|l_1⋅l_2|m1p,m2p>.
    '''
    if (l, return_matrix) in trees_dict.values:
        return trees_dict.values[(l, return_matrix)]
    jmats = Jmatrices(l, high_to_low=False)
    trees_mat = 2*sum([TensorProduct(mat1,mat1) for mat1 in jmats], sp.zeros((2*l+1)**2))
    if return_matrix:
        return trees_mat
    mbasis = mrange(l)
    trees_dictionaire = OrderedDict()
    tensor_basis = list(product(mbasis, mbasis))
    for rowidx in range(trees_mat.rows):
        row_thing = tensor_basis[rowidx]
        for colidx in range(trees_mat.cols):
            col_thing = tensor_basis[colidx]
            val = trees_mat[rowidx, colidx]
            if val !=0:
                trees_dictionaire[tuple(row_thing + col_thing)] = val
    trees_dict.values[(l, return_matrix)] = trees_dictionaire
    return trees_dictionaire
trees_dict.values = {}

def to_slater_params_fun_maker(l):
    ls = (l,l,l,l)
    def to_slater_params(qnums, coeff):
        '''
        This function will take a set of qnums that are assumed to be l1, m1, l2, m2, l1p, m1p, l2p, m2p
        and will return a set of qnums that correspond to Slater integrals and corresponding coefficients.
        '''
        l1, l2, l1p, l2p = ls
        m1, s1, m2, s2, m1p, s1p, m2p, s2p = sum(qnums, tuple())
        if not(kron(s1,s1p) and kron(s2,s2p)):
            return {}
        elif kron(m1+m2, m1p+m2p):
            if (m1 - m1p) % 2 == 0:
                phase = 1
            else:
                phase = -1
            new_dict = {}
            for k in range(6):
                key = sp.Symbol('F^{(%d)}' % k)
                c1 = (sp.sqrt((4*sp.pi) / (2*k+1)) 
                        * threeHarmonicIntegral(l1,   m1,
                                                k,   (m1 - m1p),
                                                l1p, m1p))
                c2 = (sp.sqrt((4*sp.pi) / (2*k+1)) 
                        * threeHarmonicIntegral(l2,  m2,
                                                k,   (m2-m2p),
                                                l2p, m2p))
                val = phase * coeff * c1 * c2
                if val:
                    new_dict[key] = val
            return new_dict
        else:
            return {}
    return to_slater_params


@lru_cache(maxsize=None)
def hamiltonian_CR_CF_SO_TO_Ubasis_old(num_electrons, group_label, l, sparse=False, high_to_low = False):
    '''
    Given  a  crystal field on an an ion with a given number of electrons,
    this  function  provides  the  matrix that represents a    Hamiltonian
    which includes the following interactions:

    1) the crystal field term of the given symmetry (one-body),
    2) the Coulomb repulsion between electrons (two-body),
    3) the spin-orbit interaction (one-body),
    4) and the Trees effective operator α_T L(L+1) (two-body + one-body).

    The symbols associated with each term are:

    - crystal field: B_{i,j}
    - Coulomb repulsion: F^{(k)}
    - spin-orbit interacion: \\zeta_{SO}
    - Trees operator: \\alpha_T

    The  contribution  to the total energy, as provided by the interaction
    with  the  nuclear  charge  does  not  figure  here  because it merely
    provides  a  constant  shift to all the energy levels included in this
    single-configuration approximation.

    In  all  cases  the  basis  used  in this matrix representation of the
    Hamiltonian  is  composed  of  Slater  determinants of single-electron
    states.

    The  Coulomb  repulsion appears as Slater integrals of several orders,
    and  the  crystal  field  contribution as a function of the parameters
    that parametrize it according to the corresponding symmetry group.

    The  two-electron  and  one-electron operators are evaluated using the
    Slater-Condon rules.

    The  resulting  matrix is square with size d X d with d = sp.binomial(
    2*2*l(+1),  num_electrons), which equals how many spin orbitals can be
    assigned to the given number of electrons.

    Parameters
    ----------
    num_electrons (int): how many electrons are there.
    group_label   (str): label for one of the 32 c. point groups.
    l             (int): angular momentum of pressumed ground state
    sparse (Bool): if True then the returned matrix is sp.SparseMatrix

    Returns
    -------
    (hamiltonian, basis_change, slater_dets) (tuple) with:

    hamiltonian  (sp.SparseMatrix):  given in in terms of Slater integrals
    F^{k} and the crystal field parameters adequate to the group.

    basis_change  (OrderedDict): the keys being the labels used internally
    for   calculations   and   the   values   being   Qets  understood  as
    superpositions of spherical harmonics. This for  the  single  electron
    states which are used to create multi-electron states.

    slater_dets  (list):  a  list  of  symbols  that represents the Slater
    determinants  used for the basis in which the Hamiltonian is computed,
    in  here  a  symbol with a bar on top is understood to have spin down,
    and  one  without  to have spin up (m=1/2). Together with basis_change
    this could be used to generate the multi-electron determinantal states
    in terms of Slater determinants of spherical harmonics.

    '''

    LS_dict = LSmatrix(l, HALF, high_to_low=high_to_low, as_dict=True)
    group_index = CPGs.all_group_labels.index(group_label) + 1
    cf_field = crystal_splits[group_index]
    trees_dictionaire = trees_dict(l)

    # Using uncoupled spherical harmonics basis.
    single_e_basis = [SpinOrbital(sp.Symbol('Y_{%d,%d}' % (l,m)), spin) 
                        for spin in [S_DOWN, S_UP] for m in range(-l, l+1)]
    basis_change = OrderedDict([(SpinOrbital(sp.Symbol('Y_{%d,%d}' % (l,m)), spin), Qet({(l,m,spin):1})) 
                        for spin in [S_DOWN, S_UP] for m in range(-l, l+1)])

    ham = sp.Matrix(cf_field['matrices'][0])

    def crystal_energy(qnums, coeff):
        the_dict = {1:0}
        γ1, γ2 = qnums
        γ1idx = single_e_basis.index(γ1) % ham.rows
        γ2idx = single_e_basis.index(γ2) % ham.rows
        if γ1.spin != γ2.spin:
            return {}
        else:
            the_dict[1] = coeff * ham[γ1idx, γ2idx]
            if the_dict[1] == 0:
                return {}
            else:
                return the_dict
    
    def spin_energy(qnums, coeff):
        the_dict = {1:0}
        γ1, γ2 = qnums
        m1 = list(basis_change[γ1].dict.keys())[0][1]
        m2 = list(basis_change[γ2].dict.keys())[0][1]
        s1, s2  = γ1.spin, γ2.spin
        the_dict[1] = coeff * sp.Symbol('\\zeta_{SO}') * LS_dict[((m1,s1),(m2,s2))]
        if the_dict[1] == 0:
            return {}
        else:
            return the_dict
    
    def trees_op_two_bod(qnums, coeff):
        l1, m1, s1, l2, m2, s2, l1p, m1p, s1p, l2p, m2p, s2p = qnums
        if not(kron(s1,s1p) and kron(s2,s2p)):
            return {}
        else:
            if (m1,m2,m1p,m2p) in trees_dictionaire:
                return {1: coeff * trees_dictionaire[(m1,m2,m1p,m2p)]}
            else:
                return {}
    
    def tree_op_one_bod(qnums, coeff):
        the_dict = {1:0}
        γ1, γ2 = qnums
        m1 = list(basis_change[γ1].dict.keys())[0][1]
        m2 = list(basis_change[γ2].dict.keys())[0][1]
        s1, s2  = γ1.spin, γ2.spin
        if m1 == m2 and s1 == s2:
            the_dict[1] = coeff * l * (l+1)
        if the_dict[1] == 0:
            return {}
        else:
            return the_dict
    
    # add spin up and spin down
    single_e_spin_orbitals = single_e_basis
    # create determinantal states
    slater_dets = list(combinations(single_e_spin_orbitals, num_electrons))
    slater_qets = [Qet({k:1}) for k in slater_dets]
    if sparse:
        hamiltonian = {}
    else:
        hamiltonian = []

    for idx0, qet0 in enumerate(slater_qets):
        row = []
        for idx1, qet1 in enumerate(slater_qets):
            # two electron operators
            double_braket = double_electron_braket(qet0, qet1)
            double_braket = double_braket_basis_change(double_braket, basis_change)
            coulomb_matrix_element = double_braket.apply(to_slater_params)
            coulomb_matrix_element = sp.expand(coulomb_matrix_element.as_symbol_sum())
            trees_matrix_element_twobody = (sp.Symbol('\\alpha_T')
                * sp.expand(double_braket.apply(trees_op_two_bod).as_symbol_sum()))
            # one electron operators
            single_braket = single_electron_braket(qet0, qet1)
            crystal_field__matrix_element = single_braket.apply(crystal_energy).as_symbol_sum()
            trees_matrix_element_onebod = (sp.Symbol('\\alpha_T')
                * single_braket.apply(tree_op_one_bod).as_symbol_sum())
            spinorb_energy_matrix_element = single_braket.apply(spin_energy).as_symbol_sum()
            matrix_element = (coulomb_matrix_element 
                              + trees_matrix_element_twobody
                               + trees_matrix_element_onebod
                                + crystal_field__matrix_element 
                                 + spinorb_energy_matrix_element)
            if matrix_element != 0:
                if sparse:
                    hamiltonian[(idx0,idx1)] = (matrix_element)
                else:
                    row.append(matrix_element)
            else:
                if not sparse:
                    row.append(matrix_element)

        if not sparse:
            hamiltonian.append(row)
    
    if sparse:
        hamiltonian = (sp.SparseMatrix(len(slater_qets), len(slater_qets), hamiltonian))
    else:
        hamiltonian = sp.Matrix(hamiltonian)

    return hamiltonian, basis_change, slater_dets

@lru_cache(maxsize=None)
def ham_CR(num_electrons, l, sparse=True):
    '''
    This function returns the matrix representation of the Coulomb repulsion for the
    given number of electrons with the given orbital angular momentum.

    This matrix representation is given in the multi equivalent electron basis.

    Parameters
    ----------
    num_electrons (int): 
    l             (int): orbital angular moemtnum
    sparse       (bool):

    Returns
    -------
    if sparse:
        hamiltonian (sp.matrices.sparse.MutableSparseMatrix)
    else:
        hamiltonian (sp.Matrix)
    '''
    to_slater_params = to_slater_params_fun_maker(l)
    slater_dets = elementary_basis('multi equiv electron', l, num_electrons)
    slater_qets = [Qet({k:1}) for k in slater_dets]

    if sparse:
        hamiltonian = {}
    else:
        hamiltonian = []

    for idx0, qet0 in enumerate(slater_qets):
        row = []
        for idx1, qet1 in enumerate(slater_qets):
            double_braket = double_electron_braket(qet0, qet1)
            coulomb_matrix_element = double_braket.apply(to_slater_params)
            matrix_element = sp.expand(coulomb_matrix_element.as_symbol_sum())
            if matrix_element != 0:
                if sparse:
                    hamiltonian[(idx0,idx1)] = (matrix_element)
                else:
                    row.append(matrix_element)
            else:
                if not sparse:
                    row.append(matrix_element)
        if not sparse:
            hamiltonian.append(row)
    if sparse:
        hamiltonian = (sp.SparseMatrix(len(slater_qets), len(slater_qets), hamiltonian))
    else:
        hamiltonian = sp.Matrix(hamiltonian)
    return hamiltonian

@lru_cache(maxsize=None)
def ham_CF(num_electrons, l, group_label, sparse=True):
    '''
    This function returns the matrix representation of the crystal field for the
    given number of electrons with the given orbital angular momentum.

    This matrix representation is given in the multi equivalent electron basis.

    Parameters
    ----------
    num_electrons (int): 
    l             (int): orbital angular moemtnum
    group_label   (str): label for a crystallographic point group
    sparse       (bool):

    Returns
    -------
    if sparse:
        hamiltonian (sp.matrices.sparse.MutableSparseMatrix)
    else:
        hamiltonian (sp.Matrix)
    '''
    group_index = CPGs.all_group_labels.index(group_label) + 1
    cf_matrix = sp.Matrix(crystal_splits[group_index]['matrices'][0])

    crystal_dict = {(idx0-l, idx1-l): cf_matrix[idx0, idx1] for idx0 in range(2*l+1) for idx1 in range(2*l+1)}

    slater_dets = elementary_basis('multi equiv electron', l, num_electrons)
    slater_qets = [Qet({k:1}) for k in slater_dets]

    if sparse:
        hamiltonian = {}
    else:
        hamiltonian = []

    def crystal_energy(qnums, coeff):
        the_dict = {1:0}
        ml1, ms1 = qnums[0]
        ml2, ms2 = qnums[1]
        if ms1 != ms2:
            return {}
        elif (ml1, ml2) not in crystal_dict:
            return {}
        else:
            the_dict[1] = coeff * crystal_dict[(ml1,ml2)]
            if the_dict[1] == 0:
                return {}
            else:
                return the_dict

    for idx0, qet0 in enumerate(slater_qets):
        row = []
        for idx1, qet1 in enumerate(slater_qets):
            single_braket = single_electron_braket(qet0, qet1)
            matrix_element = single_braket.apply(crystal_energy).as_symbol_sum()
            if matrix_element != 0:
                if sparse:
                    hamiltonian[(idx0,idx1)] = (matrix_element)
                else:
                    row.append(matrix_element)
            else:
                if not sparse:
                    row.append(matrix_element)
        if not sparse:
            hamiltonian.append(row)
    if sparse:
        hamiltonian = (sp.SparseMatrix(len(slater_qets), len(slater_qets), hamiltonian))
    else:
        hamiltonian = sp.Matrix(hamiltonian)
    return hamiltonian

@lru_cache(maxsize=None)
def ham_SO(num_electrons, l, sparse=True):
    '''
    This function returns the matrix representation of the spin-orbit operator
    for the given number of electrons with the given orbital angular momentum.

    This matrix representation is given in the multi equivalent electron basis.

    Parameters
    ----------
    num_electrons (int): 
    l             (int): orbital angular moemtnum
    sparse       (bool): whether returned matrix is sparse

    Returns
    -------
    if sparse:
        hamiltonian (sp.matrices.sparse.MutableSparseMatrix)
    else:
        hamiltonian (sp.Matrix)
    '''
    slater_dets = elementary_basis('multi equiv electron', l, num_electrons)
    slater_qets = [Qet({k:1}) for k in slater_dets]

    if sparse:
        hamiltonian = {}
    else:
        hamiltonian = []

    LS_dict = LSmatrix(l, HALF, as_dict=True)

    def spin_orbit_energy(qnums, coeff):
        the_dict = {1:0}
        ml1, ms1 = qnums[0]
        ml2, ms2 = qnums[1]
        the_dict[1] = coeff * sp.Symbol('\\zeta_{SO}') * LS_dict[((ml1,ms1),(ml2,ms2))]
        if the_dict[1] == 0:
            return {}
        else:
            return the_dict

    for idx0, qet0 in enumerate(slater_qets):
        row = []
        for idx1, qet1 in enumerate(slater_qets):
            single_braket = single_electron_braket(qet0, qet1)
            matrix_element = single_braket.apply(spin_orbit_energy).as_symbol_sum()
            if matrix_element != 0:
                if sparse:
                    hamiltonian[(idx0,idx1)] = (matrix_element)
                else:
                    row.append(matrix_element)
            else:
                if not sparse:
                    row.append(matrix_element)
        if not sparse:
            hamiltonian.append(row)
    if sparse:
        hamiltonian = (sp.SparseMatrix(len(slater_qets), len(slater_qets), hamiltonian))
    else:
        hamiltonian = sp.Matrix(hamiltonian)
    return hamiltonian

@lru_cache(maxsize=None)
def ham_Trees(num_electrons, l, sparse=True):
    '''
    This function returns the matrix representation of the Trees effective
    operator  for  the  given  number  of electrons with the given orbital
    angular momentum.

    This  matrix  representation is given in the multi equivalent electron
    basis.

    Parameters
    ----------
    num_electrons (int): 
    l             (int): orbital angular moemtnum
    sparse       (bool): whether returned matrix is sparse

    Returns
    -------
    if sparse:
        hamiltonian (sp.matrices.sparse.MutableSparseMatrix)
    else:
        hamiltonian (sp.Matrix)
    '''
    slater_dets = elementary_basis('multi equiv electron', l, num_electrons)
    slater_qets = [Qet({k:1}) for k in slater_dets]

    if sparse:
        hamiltonian = {}
    else:
        hamiltonian = []

    trees_dictionaire = trees_dict(l)

    def trees_op_two_bod(qnums, coeff):
        m1, s1, m2, s2, m1p, s1p, m2p, s2p = sum(qnums, tuple())
        if not(kron(s1,s1p) and kron(s2,s2p)):
            return {}
        else:
            if (m1, m2, m1p, m2p) in trees_dictionaire:
                return {1: coeff * trees_dictionaire[(m1,m2,m1p,m2p)]}
            else:
                return {}
    
    def tree_op_one_bod(qnums, coeff):
        the_dict = {1:0}
        γ1, γ2 = qnums
        ml1, ms1 = qnums[0]
        ml2, ms2 = qnums[1]
        if ml1 == ml2 and ms1 == ms2:
            the_dict[1] = coeff * l * (l+1)
        if the_dict[1] == 0:
            return {}
        else:
            return the_dict
    
    for idx0, qet0 in enumerate(slater_qets):
        row = []
        for idx1, qet1 in enumerate(slater_qets):
            double_braket = double_electron_braket(qet0, qet1)
            single_braket = single_electron_braket(qet0, qet1)
            trees_matrix_element_twobody = double_braket.apply(trees_op_two_bod)
            trees_matrix_element_twobody = (sp.Symbol('\\alpha_T')
                            * sp.expand(trees_matrix_element_twobody.as_symbol_sum()))
            trees_matrix_element_onebod  = (sp.Symbol('\\alpha_T')
                            * single_braket.apply(tree_op_one_bod).as_symbol_sum())
            matrix_element = trees_matrix_element_twobody + trees_matrix_element_onebod
            if matrix_element != 0:
                if sparse:
                    hamiltonian[(idx0,idx1)] = (matrix_element)
                else:
                    row.append(matrix_element)
            else:
                if not sparse:
                    row.append(matrix_element)
        if not sparse:
            hamiltonian.append(row)
    if sparse:
        hamiltonian = (sp.SparseMatrix(len(slater_qets), len(slater_qets), hamiltonian))
    else:
        hamiltonian = sp.Matrix(hamiltonian)
    return hamiltonian

@lru_cache(maxsize=None)
def ham_CR_CF_SO_TO(num_electrons, l, group_label, sparse=True):
    '''    
    This  function  provides  the  matrix  representation of a Hamiltonian
    which includes the following interactions:

    - CR: the Coulomb repulsion between electrons (two-body),
    - CF: the crystal field term of the given symmetry (one-body),
    - SO: the spin-orbit interaction (one-body),
    - TO: and   the   Trees effective operator α_T L(L+1) (two-body + one-
    body).

    The  basis  used  is  the  "multi equiv electron" basis as provided by
    qd.elementary_basis, which is a basis composed  of  elementary  Slater
    determinants of the form ((ml1,ms1),(ml2,ms2),...,(mln,msn)).

    The symbols associated with each term are:

    - crystal field: B_{i,j}
    - Coulomb repulsion: F^{(k)}
    - spin-orbit interacion: \\zeta_{SO}
    - Trees operator: \\alpha_T

    Since  all  these  interactions  aggregate  by  simple  addition,  the
    resulting matrix can be separated according to the  above  symbols  to
    separate the individual interactions.

    The  contribution  to the total energy, as provided by the interaction
    with  the  nuclear  charge  does  not  figure  here  because it merely
    provides  a  constant  shift to all the energy levels included in this
    single-configuration approximation.

    The  Coulomb  repulsion appears as Slater integrals of several orders,
    and  the  crystal  field  contribution as a function of the parameters
    that parametrize it according to the corresponding symmetry group.

    The  two-electron  and  one-electron operators are evaluated using the
    Slater-Condon rules.

    The  resulting  matrix is square with size d X d with d = sp.binomial(
    2*2*l(+1),  num_electrons), which equals how many spin orbitals can be
    assigned to the given number of electrons.

    Parameters
    ----------
    num_electrons (int): how many electrons are there.
    l             (int): angular momentum of pressumed ground state
    group_label   (str): label for one of the 32 c. point groups.
    sparse (Bool): if True then the returned matrix is sp.SparseMatrix

    Returns
    -------
    if sparse:
        hamiltonian (sp.MutableSparseMatrix)
    else:
        hamiltonian (sp.Matrix)

    '''
    to_slater_params = to_slater_params_fun_maker(l)
    slater_dets = elementary_basis('multi equiv electron', l, num_electrons)
    slater_qets = [Qet({k:1}) for k in slater_dets]

    if sparse:
        hamiltonian = {}
    else:
        hamiltonian = []
    
    group_index = CPGs.all_group_labels.index(group_label) + 1
    cf_matrix = sp.Matrix(crystal_splits[group_index]['matrices'][0])

    crystal_dict = {(idx0-l, idx1-l): cf_matrix[idx0, idx1] for idx0 in range(2*l+1) for idx1 in range(2*l+1)}
    
    LS_dict = LSmatrix(l, HALF, as_dict=True)
    trees_dictionaire = trees_dict(l)

    def trees_op_two_bod(qnums, coeff):
        m1, s1, m2, s2, m1p, s1p, m2p, s2p = sum(qnums, tuple())
        if not(kron(s1,s1p) and kron(s2,s2p)):
            return {}
        else:
            if (m1, m2, m1p, m2p) in trees_dictionaire:
                return {1: coeff * trees_dictionaire[(m1,m2,m1p,m2p)]}
            else:
                return {}
    
    def tree_op_one_bod(qnums, coeff):
        the_dict = {1:0}
        γ1, γ2 = qnums
        ml1, ms1 = qnums[0]
        ml2, ms2 = qnums[1]
        if ml1 == ml2 and ms1 == ms2:
            the_dict[1] = coeff * l * (l+1)
        if the_dict[1] == 0:
            return {}
        else:
            return the_dict
    def spin_orbit_energy(qnums, coeff):
        the_dict = {1:0}
        ml1, ms1 = qnums[0]
        ml2, ms2 = qnums[1]
        the_dict[1] = coeff * sp.Symbol('\\zeta_{SO}') * LS_dict[((ml1,ms1),(ml2,ms2))]
        if the_dict[1] == 0:
            return {}
        else:
            return the_dict

    def crystal_energy(qnums, coeff):
        the_dict = {1:0}
        ml1, ms1 = qnums[0]
        ml2, ms2 = qnums[1]
        if ms1 != ms2:
            return {}
        elif (ml1, ml2) not in crystal_dict:
            return {}
        else:
            the_dict[1] = coeff * crystal_dict[(ml1,ml2)]
            if the_dict[1] == 0:
                return {}
            else:
                return the_dict
    
    for idx0, qet0 in enumerate(slater_qets):
        row = []
        for idx1, qet1 in enumerate(slater_qets):
            double_braket = double_electron_braket(qet0, qet1)
            single_braket = single_electron_braket(qet0, qet1)
            cr_matrix_element = double_braket.apply(to_slater_params)
            cr_matrix_element = sp.expand(cr_matrix_element.as_symbol_sum())
            cf_matrix_element = single_braket.apply(crystal_energy).as_symbol_sum()
            so_matrix_element = single_braket.apply(spin_orbit_energy).as_symbol_sum()
            trees_matrix_element_twobody = double_braket.apply(trees_op_two_bod)
            trees_matrix_element_twobody = (sp.Symbol('\\alpha_T')
                            * sp.expand(trees_matrix_element_twobody.as_symbol_sum()))
            trees_matrix_element_onebod  = (sp.Symbol('\\alpha_T')
                            * single_braket.apply(tree_op_one_bod).as_symbol_sum())
            matrix_element = (cr_matrix_element 
                            + cf_matrix_element
                            + so_matrix_element
                            + trees_matrix_element_twobody
                            + trees_matrix_element_onebod)
            if matrix_element != 0:
                if sparse:
                    hamiltonian[(idx0,idx1)] = (matrix_element)
                else:
                    row.append(matrix_element)
            else:
                if not sparse:
                    row.append(matrix_element)
        if not sparse:
            hamiltonian.append(row)
    if sparse:
        hamiltonian = (sp.SparseMatrix(len(slater_qets), len(slater_qets), hamiltonian))
    else:
        hamiltonian = sp.Matrix(hamiltonian)
    return hamiltonian

############################## Hamiltonians ###############################
###########################################################################