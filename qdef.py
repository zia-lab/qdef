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


import os, re, pickle
import numpy as np
from math import ceil

import sympy as sp
import pandas as pd
import math
from sympy import pi, I
from sympy.physics.quantum import Ket, Bra
from sympy.physics.wigner import gaunt
from sympy.combinatorics.permutations import Permutation
from itertools import combinations, permutations, combinations_with_replacement
from functools import reduce

from collections import OrderedDict
from itertools import product, combinations
from tqdm.notebook import tqdm

from matplotlib import pyplot as plt

from IPython.display import display, HTML, Math

from misc import *
from qdefcore import *


module_dir = os.path.dirname(__file__)

# =============================================================== #
# =========================== Load others ======================= #

morrison_loc = os.path.join(module_dir,'data','morrison.pkl')
morrison = pickle.load(open(morrison_loc,'rb'))

# =========================== Load others ======================= #
# =============================================================== #


# =============================================================== #
# =======================  Ynm eval tweak ======================= #

# To avoid simplification of negative m values, the eval method
# on the spherical  harmonics  Ynm  needs  to be redefined. All
# that is done is  commenting   out  a  portion of the original
# source code.

@classmethod
def new_eval(cls, n, m, theta, phi):
    n, m, theta, phi = [sp.sympify(x) for x in (n, m, theta, phi)]
    # Handle negative index m and arguments theta, phi
    #if m.could_extract_minus_sign():
    #    m = -m
    #    return S.NegativeOne**m * exp(-2*I*m*phi) * Ynm(n, m, theta, phi)
    if theta.could_extract_minus_sign():
        theta = -theta
        return sp.Ynm(n, m, theta, phi)
    if phi.could_extract_minus_sign():
        phi = -phi
        return sp.exp(-2*I*m*phi) * sp.Ynm(n, m, theta, phi)
sp.Ynm.eval = new_eval

# =======================  Ynm eval tweak ======================= #
# =============================================================== #

# this is an ugly way of loading this
# but it's necessary given that having saved them
# as a pickle with included Qets fails to load given
# that unpickling needs knowing the class it's trying to load
# therefore I turned them into regular dictionaries
# and here they're converted to qets again

crystal_fields_raw = morrison['crystal_fields_raw']
crystal_fields = {}
for k in crystal_fields_raw:
    crystal_fields[k] = [Qet(q) for q in crystal_fields_raw[k]]

###########################################################################
#################### Calculation of Surface Harmonics #####################

def SubSupSymbol(radix,ll,mm):
    '''
    Generates a symbol placeholder for the B
    coefficients   in   the   crystal  field
    potential.
    '''
    SubSupSym = sp.symbols(r'{%s}_{%s}^{%s}' % (radix, str(ll), str(mm)))
    return SubSupSym

def SubSubSymbol(radix,ll,mm):
    '''
    Generates   a   symbol   placeholder   for  the  B
    coefficients in the crystal field potential.
    '''
    SubSubSym = sp.symbols(r'{%s}_{{%s}{%s}}' % (radix, str(ll), str(mm)))
    return SubSubSym

def kronecker(i,j):
    return 0 if i!=j else 1

def Wigner_d(l, n, m, beta):
    k_min = max([0,m-n])
    k_max = min([l-n,l+m])
    Wig_d_prefact = sp.sqrt((sp.factorial(l+n)
                          *sp.factorial(l+m)
                          *sp.factorial(l-n)
                          *sp.factorial(l-m)))
    Wig_d_summands = [((-sp.S(1))**(k - m + n)
                      * sp.cos(beta/2)**(2*l+m-n-2*k)
                      * sp.sin(beta/2)**(2*k+n-m)
                      / sp.factorial(l - n -k)
                      / sp.factorial(l + m - k)
                      / sp.factorial(k)
                      / sp.factorial(k-m+n)
                      )
                      for k in range(k_min,k_max+1)]
    Wig_d = (Wig_d_prefact*sum(Wig_d_summands)).doit()
    return Wig_d

def Wigner_D(l, n, m, alpha, beta, gamma):
    args = (l, n, m, alpha, beta, gamma)
    if args in Wigner_D.values.keys():
      return Wigner_D.values[args]
    if beta == 0:
      Wig_D = sp.exp(-I*m*alpha-I*m*gamma) * kronecker(n,m)
      if n == m:
        Wig_D = (sp.cos(-m*alpha-m*gamma)+I*sp.sin(-m*alpha-m*gamma))
      else:
        Wig_D = 0
    elif beta == pi:
      if n == -m:
        Wig_D = (-1)**l * (sp.cos(-m*alpha + m*gamma)+I*sp.sin(-m*alpha + m*gamma))
      else:
        Wig_D = 0
    else:
      Wig_D_0 = I**(abs(n)+n-abs(m)-m)
      Wig_D_1 = (sp.cos(-n*gamma-m*alpha) \
                 + I*sp.sin(-n*gamma-m*alpha)) * Wigner_d(l,n,m,beta)
      Wig_D = Wig_D_0 * Wig_D_1
      Wig_D = Wig_D
    return Wig_D
Wigner_D.values = {}

def real_or_imagined(qet):
    '''
    For  a given superposition of spherical harmonics,
    determine  if  the total has a pure imaginary (i),
    pure  real (r), or mixed character (m), it assumes
    that the coefficients in the superposition are all
    real.
    '''
    chunks = dict(qet.dict)
    valences = []
    for key in list(chunks.keys()):
        if key not in chunks.keys():
            continue
        l, m = key
        chunk = chunks[key]
        if (l,-m) in chunks:
            partner = chunks[(l,-m)]
            if abs(partner) == abs(chunk):
                if sp.sign(partner) == sp.sign(chunk):
                    if m%2 == 0:
                        valences.append("r")
                    else:
                        valences.append("i")
                else:
                    if m%2 == 0:
                        valences.append("i")
                    else:
                        valences.append("r")
            else:
                valences.append("m")
            chunks.pop((l,-m))
        else:
            valences.append("m")
        if m!=0: # if equal to zero this would have been done already
            chunks.pop(key)
    valences = list(set(valences))
    if len(valences) > 1:
        return "m"
    else:
        return valences[0]

RYlm_dict = {}
def RYlm(l, m, op_params):
    '''
    Given  a group operation (as parametrized with the
    iterable  op_params which contains Euler angles α,
    β,  γ  and  the determinant of the operation) this
    function   returns   the  effect  that  that  this
    operation has on the sperical harmonic Y_lm.

    The  result  is  a  qet  whose  keys correspond to
    values   of   (l,m)   and  whose  values  are  the
    corresponding coefficients. Keys not present imply
    that the corresponding coefficient is zero.

    Examples
    --------

    >>> print(RYlm(1,0,(pi, pi/2, pi, 1))
        Qet({(2, -2): 1/4,
             (2, -1): 1/2,
             (2, 0): sqrt(6)/4,
             (2, 1): -1/2,
             (2, 2): 1/4})

    '''
    if (l,m,*op_params) in RYlm_dict:
        return RYlm_dict[(l,m,*op_params)]
    alpha, beta, gamma, detOp = op_params
    Rf = Qet()
    for nn in range(-l,l+1):
        wigD = Wigner_D(l, m, nn, alpha, beta, gamma)
        if wigD != 0:
            Rf = Rf + Qet({(l,nn): wigD})
    ret = (sp.S(detOp)**l) * Rf
    RYlm_dict[(l,m,*op_params)] = ret
    return ret

def flatten_matrix(mah):
    '''
    A  convenience  function to flatten a sympy matrix
    into a list of lists
    '''
    return [item for sublist in mah.tolist() for item in sublist]

def trivial_sym_adapted_WF(group, l, m):
    '''
    This  returns  the  proyection  of  Y_l^m  on  the
    trivial  irreducible  representation  of the given
    group.
    '''
    if isinstance(group,str):
      group = CPGs.get_group_by_label(group)
    # Order of the group which  is  equal  to
    # the number of the elements
    order = group.order
    SALC = Qet()
    # This sum is over all elements of the group
    for op_params in group.euler_angles.values():
        SALC = SALC + RYlm(l,m,op_params)
    SALC = (sp.S(1)/order)*SALC
    SALC = SALC.apply(lambda x,y : (x, sp.simplify(y)))
    return SALC

def linearly_independent(vecs):
    '''
    Given  a  list  of vectors return the largest subset which of linearly
    independent  ones  and  the  indices  that  correspond  to them in the
    original list.
    '''
    matrix = sp.Matrix(vecs).T
    good_ones = matrix.rref()[-1]
    return good_ones, [vecs[idx] for idx in good_ones]

def trivial_irrep_basis_from_Ylm(group, l, normalize=True, verbose=False, sympathize=True):
    '''
    For  a  given  group  and  a  given  value of l, this returns a set of
    linearly  independent  symmetry adapted functions which are also real-
    valued  and  which transform as the trivial irreducible representation
    of the group.

    If  the set that is found initially contains combinations that are not
    purely  imaginary  or pure real, then the assumption is made that this
    set contains single spherical harmonics, and then sums and differences
    between m and -m are given by doing this through the values of |m| for
    the functions with mixed character.

    The  output is a list of dictionaries whose keys are (l,m) tuples, and
    whose values are the corresponding coefficients.

    Essentially  this a special case of the irrep_basis_from_Ylm function,
    with additional work added to ensure that the result is real-valued.
    '''
    # apply the projection operator on the trivial irreducible rep
    # and collect the resulting basis functions
    # together with the values of (l,m) included
    flags = []
    WFs = []
    complete_basis = []
    for m in range(-l,l+1):
        aWF = trivial_sym_adapted_WF(group, l, m)
        if len(aWF.dict)>0:
            WFs.append(aWF)
            complete_basis.extend(aWF.basis())

    complete_basis = list(sorted(list(set(complete_basis))))
    # to see if they are linearly independent
    # convert the WFs to vectors on the basis collected
    # above
    vecs = [WF.vec_in_basis(complete_basis) for WF in WFs]
    lin_indep_idx, lin_indep_vecs = linearly_independent(vecs)

    # reduce the WFs to a linearly independent set

    WFs = [WFs[i] for i in lin_indep_idx]
    # test to see if the included WFs are real, imaginary, or mixed
    # if real, keep as is
    # if purely imaginary, multiply by I
    # if mixed then collect for further processing
    realWFs = []
    mixedWFs = []
    for WF in WFs:
        valence = real_or_imagined(WF)
        if normalize:
            norm = WF.norm()
            WF = WF*(sp.S(1)/norm)
        if valence == 'r':
            realWFs.append(WF)
        elif valence == 'i':
            realWFs.append(I*WF)
        elif valence == 'm':
            flags.append('m')
            mixedWFs.append(WF)
    # collect the values of |m| included in the mixed combos
    mixedMs = set()
    if (len(mixedWFs) != 0) and verbose:
        print("\nMixtures found, unmixing...")
    for WF in mixedWFs:
        # ASSUMPTION: both m and -m are in there and only as singles
        assert len(WF.dict) == 1
        for key, val in WF.dict.items():
            mixedMs.add(abs(key[1]))
    # for the values of m in mixedMs compute the real sums
    # and differences
    for m in mixedMs:
        if m%2 == 0:
            qp = Qet({(l,m): 1}) + Qet({(l,-m): 1})
            qm = Qet({(l,m): I}) + Qet({(l,-m): -I})
            if normalize:
                qp = qp*(sp.S(1)/sp.sqrt(2))
                qm = qm*(sp.S(1)/sp.sqrt(2))
            realWFs.append(qp)
            realWFs.append(qm)
        elif m%2 == 1:
            qp = Qet({(l,m): I}) + Qet({(l,-m): I})
            qm = Qet({(l,m): 1}) + Qet({(l,-m): -1})
            if normalize:
                qp = qp*(sp.S(1)/sp.sqrt(2))
                qm = qm*(sp.S(1)/sp.sqrt(2))
            realWFs.append(qp)
            realWFs.append(qm)
    # the resulting list of realWFs must be of equal lenght
    # than WFs which in turn is equal to the number of linearly
    # independent projectd basis functions
    if len(realWFs) != len(WFs):
        raise Exception("FAILED: there are less real combos than originally")
    # in addition
    # must check that the resulting basis is still linearly independent
    # must run through the same business of collecting all the represented
    # spherical harmonics, converting that to coefficient vectors
    # and testing for linear independence
    complete_basis = []
    for WF in realWFs:
        complete_basis.extend(WF.basis())
    complete_basis = list(sorted(list(set(complete_basis))))

    vecs = [WF.vec_in_basis(complete_basis) for WF in realWFs]
    lin_indep_idx, lin_indep_vecs = linearly_independent(vecs)
    if len(lin_indep_idx) != len(WFs):
        raise Exception("FAILED: +- mixture was not faithful")
    # make the linearly independent vectors orthonormal
    lin_indep_vecs = list(map(list,sp.GramSchmidt([sp.Matrix(vec) for vec in lin_indep_vecs], normalize)))
    finalWFs = []
    if sympathize:
        better_vecs = []
        for vec in lin_indep_vecs:
            clear_elements = [abs(v) for v in vec if v!=0]
            if len(list(set(clear_elements))) == 1:
                better_vec = [0 if vl == 0 else sp.sign(vl) for vl in vec]
                better_vecs.append(better_vec)
            else:
                better_vecs.append(vec)
        lin_indep_vecs = better_vecs
    for vec in lin_indep_vecs:
        qdict = {k:v for k,v in zip(complete_basis, vec)}
        finalWFs.append(Qet(qdict))
    return finalWFs

def GramSchmidtAlt(vs, orthonormal=True):
    def projection(u,v):
        return (sp.S(u.dot(v,hermitian=True))/u.dot(u,hermitian=True))*u
    us = {}
    us[0] = vs[0]
    for k in range(1,len(vs)):
        vk = vs[k]
        projected_bit = sp.Matrix([0]*len(vs[0]))
        for j in range(k):
            projected_bit += sp.simplify(projection(us[j],vk))
        us[k] = vs[k] - projected_bit
    if orthonormal:
        es = [sp.simplify(us[k]/sp.sqrt(sp.S(us[k].dot(sp.conjugate(us[k]))))) for k in us]
        return es
    else:
        return [sp.simplify(us[k]) for k in range(len(us))]

def basis_check(group_label, irrep_symbol, qets, full_output = False):
    '''
    This   function   checks  if  a  list  of  qets  (which  are
    interpreted  as  superpositions  of Ylms for fixed l), are a
    basis  for  the  given representation. This is done by going
    through  all  the  operations  of  the  group,  applying the
    corresponding rotations to all of the components of the qets
    and  seeing  if  the  result  matches  what  is  obtained by
    directly    using    the    matrix   for   the   irreducible
    representation.
    '''
    group = CPGs.get_group_by_label(group_label)
    if not isinstance(irrep_symbol, sp.Symbol):
        irrep_symbol = sp.Symbol(irrep_symbol)
    if len(qets) != group.irrep_dims[irrep_symbol]:
        print("A basis needs to have as many entries as the size of the irrep!")
    the_one_ways = []
    irrep_way = []
    irrep_matrices = group.irrep_matrices[irrep_symbol]
    irrep_dim = group.irrep_dims[irrep_symbol]
    all_comparisons = {}
    all_checks = {}
    for R, DR in irrep_matrices.items():
        for idx, qet in enumerate(qets):
            direct_way = sum([v*RYlm(*k,group.euler_angles[R]) for k,v in qet.dict.items()],Qet())
            irrep_way = sum([DR[k,idx]*qets[k] for k in range(irrep_dim)],Qet())
        checks = []
        if set(irrep_way.dict.keys()) == set(direct_way.dict.keys()):
            for key in set(irrep_way.dict.keys()):
                v1 = sp.N(irrep_way.dict[key],chop=True)
                v2 = sp.N(direct_way.dict[key],chop=True)
                checks.append(sp.N(v1-v2,chop=True) == 0)
        else:
            check = (False)
        if sum(checks) == len(checks):
            check = True
        else:
            check = False
        all_comparisons[R] = (check,direct_way, irrep_way)
        all_checks[R] = check
    if sum(all_checks.values()) == len(all_checks):
        all_good = True
    else:
        all_good = False
    if full_output:
        return all_comparisons
    else:
        return all_good

def symmetry_adapted_basis(group_label, lmax, verbose=False):
    '''
    This function takes a  label  for a crystallographic point group and a
    maximum  value for l, and it constructs the symmetry adapted bases for
    all  of the irreducible representations of the group by taking the Ylm
    as the generating functions.

    The  result  is  a dictionary whose keys are symbols for the different
    irreducible   representations  of  the  group  and  whose  values  are
    dictionaries  whose keys are values of l and whose values are lists of
    lists  of  Qets  (in  chunks  of  length  equal  to  the  size  of the
    corresponding  irrep) that represent the linear combinations that form
    bases  that  transform  according  to  the irreducible representation.

    An  empty list means that the corresponding irreducible representation
    is not contained in the subspace for the corresponding value of l.

    --- Example ---
    symmetry_adapted_basis('O', 3) ->

    {A_1: {0: [[Qet({(0, 0): 1})]],
           1: [],
           2: [],
           3: []},
     A_2: {0: [],
           1: [],
           2: [],
           3: [[Qet({(3, -2): sqrt(2)/2,
                     (3,  2): -sqrt(2)/2})]]},
     E: {0: [],
         1: [],
         2: [[Qet({(2,  0): 1}),
              Qet({(2, -2): sqrt(2)/2,
                   (2,  2): sqrt(2)/2})]],
         3: []},
     T_1: {0: [],
           1: [[Qet({(1, -1): sqrt(2)/2,
                      (1, 1): -sqrt(2)/2}),
                Qet({(1,  0): I}),
                Qet({(1, -1): sqrt(2)*I/2,
                     (1,  1): sqrt(2)*I/2})]],
           2: [],
           3: [[Qet({(3, -3): sqrt(5)/4,
                     (3, -1): sqrt(3)/4,
                     (3,  1): -sqrt(3)/4,
                     (3,  3): -sqrt(5)/4}),
                Qet({(3,  0): -I}),
                Qet({(3, -3): -sqrt(5)*I/4,
                     (3, -1): sqrt(3)*I/4,
                     (3,  1): sqrt(3)*I/4,
                     (3,  3): -sqrt(5)*I/4})]]},
     T_2: {0: [],
           1: [],
           2: [[Qet({(2, -2): sqrt(2)/2,
                     (2,  2): -sqrt(2)/2}),
                Qet({(2, -1): sqrt(2)/2,
                     (2,  1): -sqrt(2)/2}),
                Qet({(2, -1): sqrt(2)*I/2,
                     (2,  1): sqrt(2)*I/2})]],
           3: [[Qet({(3, -2): -sqrt(2)/2,
                     (3,  2): -sqrt(2)/2}),
                Qet({(3, -3): sqrt(3)/4,
                     (3, -1): sqrt(5)/4,
                     (3,  1): sqrt(5)/4,
                     (3,  3): sqrt(3)/4}),
                Qet({(3, -3): -sqrt(3)*I/4,
                     (3, -1): sqrt(5)*I/4,
                     (3,  1): -sqrt(5)*I/4,
                     (3,  3): sqrt(3)*I/4})]]}}
    '''
    # The GramSchmidt routine from sympy fails in an odd case,
    # because of this I had to replace it with a custom version.
    GramSchmidtFun = GramSchmidtAlt
    # GramSchmidtFun = sp.GramSchmidt
    group = CPGs.get_group_by_label(group_label)
    group_irreps = group.irrep_labels
    symmetry_basis = {}
    for group_irrep in group_irreps:
        if verbose:
            print(str(group_irrep))
        irrep_dim = group.irrep_dims[group_irrep]
        symmetry_basis[group_irrep] = {}
        irrep_matrices = group.irrep_matrices[group_irrep]
        for l in range(lmax+1):
            full_basis = [(l,m) for m in range(-l,l+1)]
            all_phis = {}
            for m in range(-l,l+1):
                phis = {}
                # for a fixed row t,
                for t in range(irrep_dim):
                    # collect of of the sums by adding over columns
                    phi = Qet({})
                    for s in range(irrep_dim):
                        for R, DR in irrep_matrices.items():
                            dr = sp.conjugate(DR[t,s])
                            op_params = group.euler_angles[R]
                            if dr != 0:
                                phi = phi + dr*RYlm(l,m,op_params)
                        phis[(t,s)] = (sp.S(irrep_dim)/group.order)*phi
                all_phis[m] = phis
            # Take the qets and find the coefficients in the basis full_basis
            # this is necessary to evaluate linear independence, and useful
            # for applying the Gram-Schmidt orthonormalization process.
            coord_vecs = []
            for m,s,t in product(range(-l,l+1),range(irrep_dim),range(irrep_dim)):
                coord_vecs.append(all_phis[m][(t,s)].vec_in_basis(full_basis))
            if verbose:
                print("Constructing a big matrix with coordinates in standard basis....")
            bigmatrix = sp.Matrix(coord_vecs)
            num_lin_indep_rows = (bigmatrix.rank())
            good_rows = []
            if num_lin_indep_rows != 0:
                if verbose:
                    print("There are %d linearly independent entries..." % (num_lin_indep_rows))
                    print("Collecting that many, and in groups of %d, from the original set..." % irrep_dim)
                assert (bigmatrix.rows % irrep_dim) == 0, "No. of indep entries should divide the dim of the irrep."
                # determine what groupings have the full rank of the matrix
                chunks = [(coord_vecs[i*irrep_dim : (i*irrep_dim + irrep_dim)]) for i in range(bigmatrix.rows // irrep_dim)]
                cycles = num_lin_indep_rows // irrep_dim
                chunks = [chunk for chunk in chunks if (sp.Matrix(chunk).rank() == irrep_dim)]
                for bits in combinations(chunks,cycles):
                    mbits = list(map(lambda x: [sp.Matrix(x)], bits))
                    tot = sp.Matrix(sp.BlockMatrix(mbits))
                    if tot.rank() == num_lin_indep_rows:
                        # a satisfactory subset has been found, exit
                        break
                else:
                    raise Exception("Couldn't find an adequate subset of rows.")
                for bit in bits:
                    good_rows.append(bit)
                if verbose:
                    print("Orthonormalizing ...")
                # convert the coefficient vectors back to qets
                flat_rows = []
                degeneracy = len(good_rows)
                for rows in good_rows:
                    flat_rows.extend(rows)
                flat_rows = list(map(sp.Matrix,flat_rows))
                normalized = GramSchmidtFun(flat_rows,orthonormal=True)
                parts = []
                for deg in range(degeneracy):
                    chunk = list(map(list,normalized[deg*irrep_dim:deg*irrep_dim+irrep_dim]))
                    parts.append(chunk)
                all_normal_qets = []
                for part in parts:
                    normal_qets = [Qet({k: v for k,v in zip(full_basis,part[i]) if v!=0}) for i in range(len(part))]
                    all_normal_qets.append(normal_qets)
                if verbose:
                    print("Finished!")
                symmetry_basis[group_irrep][l] = all_normal_qets
            else:
                symmetry_basis[group_irrep][l] = []
    return symmetry_basis

generic_cf = Qet({(k,q):(sp.Symbol('B_{%d,%d}^%s' % (k,q,"r"))-sp.I*sp.Symbol('B_{%d,%d}^%s' % (k,q,"i"))) for k in [1,2,3,4,5,6] for q in range(-k,k+1)})

def compute_crystal_field(group_num):
    '''
    This  function returns a list with the possible forms that the crystal
    field  has for the given group. This list has only one element up till
    group  27,  after that the list has two possibilities that express the
    possible sign relationships between the B4q and the B6q coefficients.

    For groups 1-3 an empty list is returned.

    The  crystal  field  is  a  qet  which has as keys tuples (k,q) and as
    values sympy symbols for the corresponding coefficients.
    '''
    full_params = morrison['Bkq grid from tables 8.1-8.3 in_Morrison 1988']
    if group_num < 3:
        print("Too little symmetry, returning empty list.")
        return []
    if group_num <=27:
        cf = [generic_cf.subs(morrison['special_reps'][group_num]).subs({k:0 \
                        for k,v in full_params[group_num].items() if not v})]
    elif group_num == 28:
        cf_p = Qet({
                 (3,2): sp.Symbol('B_{3,2}^r'),
                 (3,-2): sp.Symbol('B_{3,2}^r'),
                 (4,0): sp.Symbol('B_{4,0}^r'),
                 (4,4): sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (4,-4): sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (6,0): sp.Symbol('B_{6,0}^r'),
                 (6,4): -sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 (6,-4): -sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 })
        cf_m = Qet({
                 (3,2): sp.Symbol('B_{3,2}^r'),
                 (3,-2): sp.Symbol('B_{3,2}^r'),
                 (4,0): sp.Symbol('B_{4,0}^r'),
                 (4,4): -sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (4,-4): -sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (6,0): sp.Symbol('B_{6,0}^r'),
                 (6,4): sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 (6,-4): sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 })
        cf = [cf_p, cf_m]
    elif group_num == 29:
        cf_p = Qet({
                 (4,0): sp.Symbol('B_{4,0}^r'),
                 (4,4): sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (4,-4): sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (6,0): sp.Symbol('B_{6,0}^r'),
                 (6,4): -sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 (6,-4): -sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 })
        cf_m = Qet({
                 (4,0): sp.Symbol('B_{4,0}^r'),
                 (4,4): -sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (4,-4): -sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (6,0): sp.Symbol('B_{6,0}^r'),
                 (6,4): sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 (6,-4): sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 })
        cf = [cf_p, cf_m]
    elif group_num == 30:
        cf = compute_crystal_field(29)
    elif group_num == 31:
        cf = compute_crystal_field(28)
    elif group_num == 32:
        cf = compute_crystal_field(30)
    return cf

#################### Calculation of Surface Harmonics #####################
###########################################################################

###########################################################################
####################### Irreducible basis transforms ######################

def irrepqet_to_lmqet(group_label, irrep_qet, l, returnbasis=False):
    '''
    This  function  take  a  label  for a crystallographic point
    group  and  a  qet  written in an irreducible representation
    basis,  understood  to  originate from the given value of l,
    and  it  returns  the  representation  in the standard (l,m)
    basis.

    Parameters
    ----------

    group_label : str
                A  label for crystallographic point group, must be
                one of CGPs.all_group_labels
    irrep_qet   : Qet
                A  Qet  whose  keys  are  assumed labels for irrep
                components    and   whose   values   are   complex
                coefficients.
    returnbasis : bool
                For debugging purposes only. If True, the function
                also  returns  the  list  whose  elements  are the
                coefficients  of  the  basis  vectors  used in the
                decomposition.

    Returns
    -------
    tuple : (l, p_qet)

        l       : int
                The value of l for the input Qet.
       lm_qet   : Qet
                A  Qet  whose  keys  are  (l,m) tuples and whose
                values are complex coefficients.

    See Also
    --------
    lmqet_to_irrepqet

    Examples
    --------
    import sympy as sp

    >>> irrep_qet = Qet({sp.Symbol('u_{E}'):sp.sqrt(2)/2, sp.Symbol('y_{T_2}'):-sp.sqrt(2)/2})
    >>> print(irrepqet_to_lmqet('O', irrep_qet, 2))
        (2,
        Qet({(2,-1):-1/2,
             (2,0) :sqrt(2)/2,
             (2,1) :1/2})
        )
    '''
    group = CPGs.get_group_by_label(group_label)
    irrep_basis = group.symmetry_adapted_bases
    irrep_labels = group.irrep_labels
    l_basis = [(l,m) for m in range(-l,l+1)]
    # pick out all the parts for this value of l
    this_irrep_basis = {ir: irrep_basis[ir][l] for ir in irrep_labels}
    # kick out the ones that are empty
    this_irrep_basis = {k:v for k,v in this_irrep_basis.items() if len(v) != 0}
    size_of_basis = {k:len(v) for k, v in this_irrep_basis.items()}
    # convert the qets in the bases to coordinate vectors in the standard basis
    this_irrep_basis = {(irrep_label,basis_idx):list( \
                    map(lambda x: x.vec_in_basis(l_basis), basis)) \
                       for irrep_label, bases in this_irrep_basis.items() \
                       for basis_idx, basis in enumerate(bases)}
    # flatten and grab component labels
    nice_basis = {}

    for k, basis in this_irrep_basis.items():
        irrep_label = k[0]
        basis_idx = k[1]
        irrep_dim = group.irrep_dims[irrep_label]
        num_basis_vecs =  size_of_basis[irrep_label]
        if num_basis_vecs == 1:
            basis_labels = [((group.component_labels[irrep_label][i])) \
                                            for i in range((irrep_dim))]
        else:
            basis_labels = [(sp.Symbol(("{}^{(%d)}%s") % (basis_idx+1, \
                        str(group.component_labels[irrep_label][i])))) \
                                            for i in range((irrep_dim))]
        nice_basis.update(OrderedDict(zip(basis_labels, basis)))
    lm_qet = [sp.Matrix(nice_basis[k])*v for k,v in irrep_qet.dict.items()]
    lm_qet = sum(lm_qet, sp.Matrix([0]*(2*l+1)))
    lm_qet = Qet(dict(zip(l_basis,lm_qet)))
    if returnbasis:
        return l, lm_qet, list(nice_basis.values())
    else:
        return l, lm_qet


def lmqet_to_irrepqet(group_label, qet, returnbasis=False):
    '''
    Given  a qet whose keys are (l,m) values, for a shared value
    of l this function determines the representation in terms of
    irreducible basis functions.

    Parameters
    ----------

    group_label : str
                A  label for crystallographic point group, must be
                one of CGPs.all_group_labels
    qet         : Qet
                A  Qet  whose  keys are assumed to be tuples (l,m)
                and whose values are complex numbers.
    returnbasis : bool
                If  True, the function also returns the list whose
                elements are the coefficients of the basis vectors
                used  in  the  decomposition.

    Returns
    -------

    tuple : (l, p_qet)

        l       : int
                The value of l for the input Qet.
        p_qet   : Qet
                A  Qet  whose  keys  are  labels  to components of
                irreducible   reps   and   whose  values  are  the
                corresponding coefficients. If the component label
                is  accompanied  by  a  left-superindex  then that
                number  in  parenthesis  is  indexing which of the
                basis this component is referring to.

    See Also
    --------
    irrepqet_to_lmqet

    Examples
    --------

    >>> qet = Qet({(2,-1):-1/2, (2,0):1/2, (2,1):1/2})
    >>> print(lmqet_to_irrepqet('O', qet))
        (2,
        Qet({u_{E}  : 1/2,
             y_{T_2}: -sqrt(2)/2}
           )
        )
    '''
    group = CPGs.get_group_by_label(group_label)
    irrep_basis = group.symmetry_adapted_bases
    irrep_labels = group.irrep_labels
    # for brevity I will only consider the case where there is a single l represented
    # in the qet
    ls = list(set((map(lambda x: x[0],qet.dict.keys()))))
    assert len(ls) == 1, "qet must be a supersposition of equal values of l"
    l = ls[0]
    l_basis = [(l,m) for m in range(-l,l+1)]
    # pick out all the parts for this value of l
    this_irrep_basis = {ir: irrep_basis[ir][l] for ir in irrep_labels}
    # kick out the ones that are empty
    this_irrep_basis = {k:v for k,v in this_irrep_basis.items() if len(v) != 0}
    size_of_basis = {k:len(v) for k, v in this_irrep_basis.items()}
    # convert the qets in the bases to coordinate vectors in the standard basis
    this_irrep_basis = {(irrep_label,basis_idx):list(map(lambda x: x.vec_in_basis(l_basis), basis)) \
                   for irrep_label, bases in this_irrep_basis.items() \
                   for basis_idx, basis in enumerate(bases)}
    # flatten and grab component labels
    nice_basis = {}

    for k, basis in this_irrep_basis.items():
        irrep_label = k[0]
        basis_idx = k[1]
        irrep_dim = group.irrep_dims[irrep_label]
        num_basis_vecs =  size_of_basis[irrep_label]
        if num_basis_vecs == 1:
            basis_labels = [(irrep_label,i,basis_idx,(group.component_labels[irrep_label][i])) for i in range((irrep_dim))]
        else:
            basis_labels = [(irrep_label,i,basis_idx,sp.Symbol(("{}^{(%d)}%s") % (basis_idx+1,str(group.component_labels[irrep_label][i])))) for i in range((irrep_dim))]
        nice_basis.update(OrderedDict(zip(basis_labels, basis)))
    # finally, proyect the qet onto the basis
    proyected_qet = {}
    c_qet = qet.vec_in_basis(l_basis)
    p_qet = {k:sp.Matrix(c_qet).dot(sp.Matrix(v)) for k,v in nice_basis.items()}
    # clean out bits that are zero, and convert to Qet
    p_qet = Qet({k[-1]: v for k,v in p_qet.items() if v!=0})
    if returnbasis:
        return l, p_qet, list(nice_basis.values())
    else:
        return l, p_qet

####################### Irreducible basis transforms ######################
###########################################################################


###########################################################################
############### Calculation of Clebsch-Gordan Coefficients ################

def cg_symbol(comp_1, comp_2, irep_3, comp_3):
    '''
    Given  symbols  for three components (comp_1, comp_2, comp_3) of three
    irreducible  representations  of  a  group  this  function  returns  a
    sp.Symbol for the corresponding Clebsch-Gordan coefficient:

    <comp_1,comp_2|irep_3, comp_3>

    The symbol of the third irrep is given explicitly as irep_3. The other
    two  irreps are implicit in (comp_1) and (comp_2) and should be one of
    the symbols in group.irrep_labels.

    (comp_1,   comp_2,   and   comp_3)  may  be  taken  as  elements  from
    group.component_labels.
    '''
    symb_args = (comp_1, comp_2, irep_3, comp_3)
    return sp.Symbol(r"{\langle}%s,%s|%s,%s{\rangle}" % symb_args)

class V_coefficients_old():
    '''
    This class loads data for the V coefficients for the octahedral group
    as defined in Appendix C of Griffith's book "The  Irreducible  Tensor
    Method for Molecular Symmetry Groups".
    In here the labels for the components for the irreducible  reps  have
    been matched in the following way:
    A_1 : \iota -> a_{A_1}
    A_2 : \iota -> a_{A_2}
    E   : \theta -> u_{E} \epsilon -> v_{E}
    T_1 : x -> x_{T_1} y -> y_{T_1} z -> z_{T_1}
    T_2 : x -> x_{T_2} y -> y_{T_2} z -> z_{T_2}
    '''
    def __init__(self):
        self.coeffs = pickle.load(open(vcoeffs_fname,'rb'))
    def eval(self, args):
        '''
        Args must be a tuple of sympy symbols for irreducible representations
        and components. This  tuple  must  contain  interleaved  symbols  for
        irreducible representations and components.
        args must match the template (a,α,b,β,c,γ) that matches with
        Griffith's notation like so:
                        V ⎛ a b c ⎞
                          ⎝ α β γ ⎠.
        '''
        return self.coeffs[args]

class V_coefficients():
    vcoeffs_fname = os.path.join(module_dir,'data','Vcoeffs_vanilla.pkl')
    @classmethod
    def vanilla(cls):
        '''
        This  function  loads  data  for  the V coefficients for the
        octahedral group as defined in Appendix C of Griffith's book
        "The   Irreducible  Tensor  Method  for  Molecular  Symmetry
        Groups".

        In  here  the  labels for the components for the irreducible
        reps  have  been  matched  in  the  following  way.  This is
        probably wrong.

        A_1 : \iota -> a_{A_1}
        A_2 : \iota -> a_{A_2}
        E   : \theta -> u_{E} \epsilon -> v_{E}
        T_1 : x -> x_{T_1} y -> y_{T_1} z -> z_{T_1}
        T_2 : x -> x_{T_2} y -> y_{T_2} z -> z_{T_2}
        '''
        return pickle.load(open(cls.vcoeffs_fname,'rb'))
    @classmethod
    def type_of_combo(cls, irrep_combo, V_coeff, group):
        '''
        Given  a  triple of irreducible representation symbols and a
        given  set of V coefficients, this function determines which
        type  of  symmetry  the  given  coefficients  have  for that
        triple.
        It does this by evaluating and comparing permutations of the
        arguments for the given V_coeff. If all values are zero, the
        triple  is  singular, if all permutations result in the same
        value, then it has even symmetry, if all signed-permutations
        result  in  the same value, then it has odd symmetry, and if
        none of these apply the it is 'neither'.


        Parameters
        ----------
        irrep_combo (iterable):  three sp.Symbol for three irreps.
        V_coeff        (dict) :
                        keys are 6-tuples where the first three values are
                        symbols  for  irreducible  representations and the
                        final   three   are   symbols   for  corresponding
                        components.
        group  (qdefcore.CrystalGroup): a crystal group.

        Returns
        -------
        (dict): {
                type (str) : one of 'singular', 'even', 'odd', 'neither'
                even_comparisons (dict):
                        Keys   are   irrep   triples   (a  permutation  of
                        irrep_combo)  and  values are lists of OrderedDict
                        whose  keys  are  V_arg  and  whose values are the
                        corresponding coefficients.
                odd_comparisons (dict):
                        keys   are   irrep   triples   (a  permutation  of
                        irrep_combo)   and   whose  values  are  lists  of
                        OrderedDict  whose keys are V_arg and whose values
                        are  the  corresponding coefficients multiplied by
                        the sign of the corresponding permutation.
        }
        '''
        trio_types = ['singular', 'even', 'odd', 'neither']
        irrep_combo = tuple(irrep_combo)
        # grab the labels for the components of the given irreps
        all_components = [group.component_labels[ir] for ir in irrep_combo]
        standard_permutation_signs = [(1-2*Permutation(x).parity()) for x in list(permutations([0,1,2]))]
        all_values = []
        even_chunks, even_checks = [], []
        odd_chunks, odd_checks = [], []
        # collect all the values in one big list
        # useful to determine if the given irrep combo is singular
        for components in product(*all_components):
            columns = list(zip(irrep_combo, components))
            column_permutations = list(permutations(columns))
            for column_permutation in column_permutations:
                arg_0, arg_1 = list(zip(*column_permutation))
                V_arg = tuple(arg_0 + arg_1)
                all_values.append(V_coeff[V_arg])
        # go across all possible arguments of the V_coefficients
        for components in product(*all_components):
            columns = list(zip(irrep_combo, components))
            column_permutations = list(permutations(columns))
            signed_permutted_values = []
            permutted_values = []
            permutation_signs = list(standard_permutation_signs)
            even_sector = OrderedDict()
            odd_sector = OrderedDict()
            # permutations is agnostic to the columns being equal
            # if there are two columns or more that are equal
            # ther permutation signs need to be coerced all to be
            # zero
            if len(set(columns)) < 3:
                    permutation_signs = [1 for x in list(permutations([0,1,2]))]
            # evaluate the coefficients across permutation of columns
            # save each comparison in its corresponding dictionary,
            # either even_sector or odd_sector
            for column_permutation, perm_sign in zip(column_permutations, permutation_signs):
                arg_0, arg_1 = list(zip(*column_permutation))
                V_arg = tuple(arg_0 + arg_1)
                signed_permutted_values.append(perm_sign*V_coeff[V_arg])
                permutted_values.append(V_coeff[V_arg])
                all_values.append(V_coeff[V_arg])
                even_sector[V_arg] = V_coeff[V_arg]
                odd_sector[V_arg] = perm_sign*V_coeff[V_arg]
            odd_chunks.append(odd_sector)
            even_chunks.append(even_sector)
            # if only one value remains that means that all
            # the permutted values have the corresponding symmetry
            odd_checks.append(len(set(signed_permutted_values)) == 1)
            even_checks.append(len(set(permutted_values)) == 1)
        all_values = list(set(all_values))
        is_singular = (len(all_values) == 1 and all_values[0] == 0)
        is_even = (all(even_checks) and not is_singular)
        is_odd = (all(odd_checks) and not is_singular)
        # in some cases odd and even are exactly the same
        # this for example when all irreps are equal
        # if this occurs, then it is defined as even
        if is_even and is_odd:
            is_odd = False
        is_neither = (not is_singular) and (not is_even) and (not is_odd)
        # at most it is in one of these categories
        assert sum([is_singular, is_even, is_odd, is_neither]) == 1
        true_index = [is_singular, is_even, is_odd, is_neither].index(True)
        the_type = trio_types[true_index]
        return {'type': the_type, 'even_comparisons': even_chunks, 'odd_comparisons': odd_chunks}
    @classmethod
    def V_fixer(cls, V_coeff, sign_changes, group, verbose=False):
        '''
        This  function takes a dictionary of sign changes, and a dictionary of
        V_coefficients.  After  implementing  the  sign  changes  on the given
        coefficient   it   then   determines   which  triples  of  irreducible
        representations are even, which are odd, and which are neither.

        Parameters
        ----------

        V_coeff      (dict):
                            all V coefficients for the current group.
        sign_changes (dict):
                            keys  are  triples  of  irreps  and values are the
                            associated  multiplier  such  that  every  V_coeff
                            whose  top  row  is  the key, is multiplied by the
                            given  multiplier. In principle these values could
                            only  be negative, in practice having values which
                            are +1 may be good for accounting.
        verbose.     (bool):
                            Print or not progress messages.

        Returns
        -------

        (dict):
            {'V_coeff'     (dict):
                            The  V  coefficients  as changed by the given sign
                            changes.
             'combo_types' (dict):
                            Keys equal triples of irreps values are type which
                            it is.
             'all_permutation_comparison' (dict):
                            For  each outer key ('even' or 'odd') this gives a
                            dictionary  whose keys are irrep triples and whose
                            values  are  lists  of  OrderedDict whose keys are
                            V_arg  and  whose  values  are  the  corresponding
                            coefficients.   These  are  useful  to  track  the
                            evaluation of the type of each irrep triple.
                             }
        '''
        if verbose: print("Performing requested changes of sign ...")
        for V_arg in V_coeff.keys():
            head = V_arg[:3]
            if head in sign_changes.keys():
                V_coeff[V_arg] = V_coeff[V_arg]*sign_changes[head]
        if verbose: print("Determining triples of irreducible representations ...")
        irrep_combos = list(combinations_with_replacement(group.irrep_labels,3))
        if verbose: print("Making comparisons across permutations of columns...")
        all_permutation_comparisons = {'even': {}, 'odd': {}}
        combo_types = {}
        for irrep_combo in irrep_combos:
            combify = cls.type_of_combo(irrep_combo, V_coeff, group)
            combo_types[irrep_combo] = combify['type']
            all_permutation_comparisons['even'][irrep_combo] = combify['even_comparisons']
            all_permutation_comparisons['odd'][irrep_combo] = combify['odd_comparisons']
        return {'V_coeff': V_coeff, 'combo_types': combo_types, 'all_permutation_comparisons':all_permutation_comparisons}

def group_clebsch_gordan_coeffs(group, Γ1, Γ2, rep_rules = True, verbose=False):
    '''
    Given   a  group  and  symbol  labels  for  two  irreducible
    representations  Γ1  and  Γ2  this  function  calculates the
    Clebsh-Gordan  coefficients used to span the basis functions
    of  their  product  in terms of the basis functions of their
    factors.

    By  assuming  an  even phase convention for all coefficients
    the result is also given for the exchanged order (Γ2, Γ1).

    If  rep_rules  = False, this function returns a tuple with 3
    elements,  the  first  element being a matrix of symbols for
    the  CGs  coefficients  for  (Γ1, Γ2) the second element the
    matrix  for  symbols  for (Γ2, Γ1) and the third one being a
    matrix  to which its elements are matched element by element
    to the first and second matrices of symbols.

    If rep_rules = True, this function returns two dictionaries.
    The  keys  in the first one equal CGs coefficients from (Γ1,
    Γ2)  and  the second one those for (Γ2, Γ1); with the values
    being the co- rresponding coefficients.

    These  CG  symbols  are  made  according  to  the  following
    template:
        <i1,i2|i3,i4>
        (i1 -> symbol for basis function in Γ1 or Γ2)
        (i2 -> symbol for basis function in Γ2 or Γ1)
        (i3 -> symbol for an ir rep Γ3 in the group)
        (i4 -> symbol for a basis function of Γ3)
    '''
    irreps = group.irrep_labels
    irep1, irep2 = Γ1, Γ2
    # must first find the resulting direct sum decomposition of their product
    irep3s = group.product_table.odict[(irep1, irep2)]
    if irep3s == []: # this is needed in case theres a single term in the direct sum
        irep3s = [group.product_table.odict[(irep1, irep2)]]
    # also need to grab the labels for a set of generators
    generators = group.generators
    component_labels = group.component_labels
    print("Grabbing the labels for the basis functions ...") if verbose else None
    labels_1, labels_2 = component_labels[irep1], component_labels[irep2]
    cg_size = len(labels_1)*len(labels_2)
    print("CG is a ({size},{size}) matrix ...".format(size=cg_size)) if verbose else None
    generators_1 = [group.irrep_matrices[irep1][g] for g in generators]
    generators_2 = [group.irrep_matrices[irep2][g] for g in generators]

    # then create the list of linear constraints
    print("Creating the set of linear constraints ...") if verbose else None
    # In (2.31) there are five quantities that determine one constraints
    all_eqns = []
    for irep3 in irep3s:
        labels_3 = component_labels[irep3]
        for generator in generators:
            D1, D2, D3 = [group.irrep_matrices[irep][generator] for irep in [irep1,irep2,irep3]]
            γ1s, γ2s, γ3s = [list(range(D.shape[0])) for D in [D1,D2,D3]]
            for γ1, γ2, γ3p in product(γ1s, γ2s, γ3s):
                lhs = []
                for γ1p in γ1s:
                    for γ2p in γ2s:
                        symb_args = (labels_1[γ1p],labels_2[γ2p],irep3,labels_3[γ3p])
                        chevron = cg_symbol(*symb_args)
                        coeff = D1[γ1, γ1p]*D2[γ2,γ2p]
                        if coeff:
                            lhs.append(coeff*chevron)
                lhs = sum(lhs)
                rhs = []
                for γ3 in γ3s:
                    symb_args = (labels_1[γ1],labels_2[γ2],irep3,labels_3[γ3])
                    chevron = cg_symbol(*symb_args)
                    coeff = D3[γ3, γ3p]
                    if coeff:
                        rhs.append(coeff*chevron)
                rhs = sum(rhs)
                eqn = rhs-lhs
                if (eqn not in all_eqns) and (-eqn not in all_eqns) and (eqn != 0):
                    all_eqns.append(eqn)

    # collect all the symbols included in all_eqns
    free_symbols = set()
    for eqn in all_eqns:
        free_symbols.update(eqn.free_symbols)
    free_symbols = list(free_symbols)

    # convert to matrix of coefficients
    coef_matrix = sp.Matrix([[eqn.coeff(cg) for cg in free_symbols] for eqn in all_eqns])
    # and simplify using the rref
    rref = coef_matrix.rref()[0]
    # turn back to symbolic and solve
    better_eqns = [r for r in rref*sp.Matrix(free_symbols) if r!=0]
    # return better_eqns, free_symbols
    try:
        better_sol = sp.solve(better_eqns, free_symbols)
    except:
        better_sol = sp.solve(better_eqns, free_symbols, manual=True)
    # construct the unitary matrix with all the CGs
    print("Building the big bad matrix ...") if verbose else None
    U_0 = []
    U_1 = []
    for γ1 in labels_1:
        for γ2 in labels_2:
            row_0 = []
            row_1 = []
            for irep3 in irep3s:
                labels_3 = component_labels[irep3]
                for γ3 in labels_3:
                    # the given order and the exchanged one
                    # is saved here to take care of the phase
                    # convention upon exchange of Γ1 and Γ2
                    chevron_0 = cg_symbol(γ1, γ2, irep3, γ3)
                    chevron_1 = cg_symbol(γ2, γ1, irep3, γ3)
                    row_0.append(chevron_0)
                    row_1.append(chevron_1)
            U_0.append(row_0)
            U_1.append(row_1)
    Usymbols_0 = sp.Matrix(U_0)
    Usymbols_1 = sp.Matrix(U_1)
    # replace with the solution to the linear constraints
    U_0 = Usymbols_0.subs(better_sol)
    # build the unitary constraints
    print("Bulding the unitarity constraints and assuming U to be orthogonal ...") if verbose else None
    unitary_constraints = U_0*U_0.T - sp.eye(U_0.shape[0])
    # flatten and pick the nontrivial ones
    unitary_set = [f for f in flatten_matrix(unitary_constraints) if f!=0]
    # solve
    try:
        unitary_sol = sp.solve(unitary_set)
    except:
        unitary_sol = sp.solve(unitary_set, manual=True)
    print("%d solutions found ..." % len(unitary_sol)) if verbose else None
    Usols = [U_0.subs(sol) for sol in unitary_sol]
    sol_0 = Usols[0]
    if not rep_rules:
        return Usymbols_0, Usymbols_1, sol_0
    else:
        dic_0 = {k:v for k,v in zip(list(Usymbols_0), list(sol_0))}
        dic_1 = {k:v for k,v in zip(list(Usymbols_1), list(sol_0))}
        return dic_0, dic_1

############### Calculation of Clebsch-Gordan Coefficients ################
###########################################################################

###########################################################################
################################## Others #################################

def mbasis(l):
    '''
    Return a row matrix with symbols corresponding to the kets
    that span the angular momentum space for a given value of l.
    '''
    return sp.Matrix([[sp.Symbol('|%d,%d\\rangle' % (l,ml)) for ml in range(-l,l+1)]])

def lmbasis(lmax):
    '''
    Return a dictionary whose keys correspond to (l,m) up till l=lmax
    and whose values are coordinate vectors corresponding to them.
    '''


def fmt_table(data, center_data=False, add_row_nums=False):
    '''Create a LaTeX table from a given list of lists'''
    buf='''
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{%s}
'''
    max_cols = max(len(r) for r in data)
    column_spec = '|' + '|'.join(['c']*max_cols) + '|'
    buf = buf % column_spec
    row_idx = 0
    for row_data in data:
        row = ''
        if add_row_nums and row_idx > 0:
            row += str(row_idx) + " & "
        if center_data and row_idx > 0:
            to_add = ceil( (max_cols - len(row_data))/2 )
            row += ' & '.join([''] * to_add)
        row += ' & '.join([sp.latex(thing) for thing in row_data])
        if row_idx == 0:
            row = '''\\hline ''' + row + '''\\\\\hline '''
        else:
            row += '''\\\\\hline '''
        row += "\n"
        buf +=row
        row_idx += 1
    buf += '''\\end{array}'''
    return buf


def threeHarmonicIntegral(l1, m1, l2, m2, l3, m3):
    '''
    Returns the value of the three spherical harmonics integral,
    the variety with the first one having a complex conjugate.

    - It may be non-zero only if l1 + l2 + l2 is even.
    - It is non-zero unless |l1-l2| <= l3 <= l1+l2

    .. math:: \int d\Omega Y_{l_1,m_1}^* Y_{l_2,m_2} Y_{l_3,m_3}

    '''
    return sp.S(-1)**m1 * gaunt(l1,l2,l3,-m1,m2,m3)


def l_splitter(group_num_or_label, l):
    '''
    This  function  takes  a  value of l and determines how many
    times  each  of the irreducible representations of the given
    group  is contained in the reducible representation obtained
    from   the   irreducible  representation  of  the  continous
    rotation group as represented by the set of Y_{l,m}.

    More simply stated it returns how states that transform like
    a  given  l  would split into states that would transform as
    the irreducible representations of the given group.

    The   function  returns  a  Qet  whose  keys  correspond  to
    irreducible  representation  symbols  of the given group and
    whose  values  are  how many times they are contained in the
    reducible representation of the group.

    Parameters
    ----------

    group_num_or_label  :  int  or  str  Index  or  string for a
    crystallographic  point  group.  l  :  int  or  str azimutal
    quantum number (arbitrary) or string character s,p,d,f,g,h,i

    Examples
    --------

    A  d-electron  in  an  cubic  field  splits into states that
    transform  either  like  E or like T_2. This can be computed
    with this function like so:

    >>> print(l_splitter('O', 2))
        Qet({E: 1, T_2: 1})

    An if one were interested in how the states of an f-electron
    split under C_4v symmetry, one could

    >>> print(l_splitter('C_{4v}',  3))
        Qet({A_2: 1, B_2: 1, B_1: 1, E:2})

    '''
    if isinstance(l, str):
        l = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}[l]
    l = sp.S(l)
    if isinstance(group_num_or_label, str):
        group_num = CPGs.all_group_labels.index(group_num_or_label)+1
    else:
        group_num = group_num_or_label
    group = CPGs.groups[group_num]
    charVec = []
    # calculate the character of each of the classes
    # in the reducible representation for the given l
    # this requires iterating over all the classes
    # and picking up any of its symmetry operations
    # figuring out what is the angle of rotation
    # that corresponds to the rotation part of
    # the matrix that represents it
    for group_class in group.classes:
        group_op = group.classes[group_class][0]
        _, _, _, detOp = group.euler_angles[group_op]
        rot_matrix = group.operations_matrices[group_op]*detOp
        cos_eta = (rot_matrix.trace()-1)/sp.S(2)
        eta = sp.acos(cos_eta)
        if sp.N(eta,chop=True) == 0:
            char = 2*l + 1
        else:
            char = sp.sin((l+sp.S(1)/2)*eta)/sp.sin(eta/2)
        charVec.append(char)
    # Once the characters have been computed
    # the inverse of the matrix representing the character
    # table for the group can be used.
    charVec = sp.N(sp.Matrix(charVec),chop=True)
    splitted = list(map(lambda x: round(sp.N(x,chop=True)),
                list(group.character_table_inverse*charVec)))
    return Qet(dict(zip(group.irrep_labels,splitted)))

def Bsimple(Bexpr):
    '''
    Takes   a   sympy   expression,   finds   the  Bnm
    coefficients  in  it and if there's only a real or
    an  imaginary  part then it returns the expression
    without the ^r or ^i.
    '''
    free_symbs = list(Bexpr.free_symbols)
    symb_counter = {}
    for symb in free_symbs:
        if 'B_' not in str(symb):
            continue
        kqcombo = eval(str(symb).split('{')[-1].split('}')[0])
        reorim = str(symb).split('^')[-1]
        if kqcombo not in symb_counter.keys():
            symb_counter[kqcombo] = 1
        else:
            symb_counter[kqcombo] += 1
    simpler_reps = {}
    for k, v in symb_counter.items():
        if v == 1:
            only = sp.Symbol('B_{%d,%d}' %k)
            real = sp.Symbol('B_{%d,%d}^r' %k)
            imag = sp.Symbol('B_{%d,%d}^i' %k)
            simpler_reps[real] = only
            simpler_reps[imag] = only
    return Bexpr.subs(simpler_reps, simultaneous=True)

regen_fname = os.path.join(module_dir,'data','CPGs.pkl')
crystal_fields_fname = os.path.join(module_dir,'data','CPGs.pkl')

if os.path.exists(regen_fname):
    print("Reloading %s ..." % regen_fname)
    CPGs = pickle.load(open(os.path.join(module_dir,'data','CPGs.pkl'),'rb'))
else:
    print("%s not found, regenerating ..." % regen_fname)
    CPGs = CPGroups(group_data)
    pickle.dump(CPGs, open(os.path.join(module_dir,'data','CPGs.pkl'),'wb'))

################################## Others #################################
###########################################################################
