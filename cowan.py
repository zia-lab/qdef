#!/usr/bin/env python

######################################################################
#                     ______                                         #
#                    / ____/___ _      ______ _____                  #
#                   / /   / __ \ | /| / / __ `/ __ \                 #
#                  / /___/ /_/ / |/ |/ / /_/ / / / /                 #
#                  \____/\____/|__/|__/\__,_/_/ /_/                  #
#                                                                    #
#                                                                    #
######################################################################

from sympy.physics.wigner import clebsch_gordan
import numpy as np
import sympy as sp
import networkx as nx
from qdefcore import *
from collections import namedtuple, OrderedDict
from itertools import product
from functools import reduce


class cg():
    remember = {}

    @classmethod
    def eva(cls, *args):
        if args in cls.remember.keys():
            return cls.remember[args]
        else:
            acg = clebsch_gordan(*args)
            cls.remember[args] = acg
            return acg


def addtwo(j1, j2):
    '''
    This  function  takes  two  angular  momenta  j1, and j2 and returns a
    dictionary  whose  keys  are  (J,mJ)  tuples, and whose values are the
    linear  combinations  of  (mj1,mj2)  that  add up to the total angular
    momentum ket.'''
    Jkets = {}
    for J in np.arange(np.abs(j1-j2), j1+j2+1, 1):
        J = int(J)
        for mJ in range(-J, J+1, 1):
            kets = [cg.eva(j1, j2, J, m1, mJ-m1, mJ)*Ket((m1, mJ-m1))
                    for m1 in np.arange(-j1, j1+1, 1)]
            Jkets[(J, mJ)] = sum(kets)
    return Jkets


def block_form(matrix):
    '''
    This  function  takes  a matrix and rearranges its columns and rows so
    that  it  is  in  block  diagonal form. It returns a tuple whose first
    element  is  the  reorganized  matrix,  whose  second  element  is the
    reordering  of  the rows, and whose third element is the reordering of
    the columns.

    As of now, it is a little ad-hoc and may fail horribly.

    If this has a solution, there's many of them, this function would give
    one of those.

    Example
    -------

    >>> test_matrix = sp.Matrix(sp.BlockDiagMatrix(*[sp.randMatrix(s) for s in [3,1,2]]))
    >>> size = test_matrix.rows
    >>> num_shuffles = 20
    >>> for _ in range(num_shuffles):
    >>>     direction = randint(0,1)
    >>>     col1 = randint(0,size-1)
    >>>     col2 = randint(0,size-1)
    >>>     permutation = list(range(size))
    >>>     permutation[col1], permutation[col2] = col2, col1
    >>>     test_matrix = test_matrix.permute(permutation,orientation=['cols','rows'][direction])
    >>> display(test_matrix)
    >>> display(block_form(test_matrix)[0])


    '''
    matrix = sp.Matrix(matrix)
    connectome = []
    # when this is called, sympy lets go of the elements that are zero
    nonz = list(matrix.todok().keys())
    for node in nonz:
        connectome.extend([(node, k) for k in nonz if
                           ((k[0] == node[0] or k[1] == node[1]))])
    matrixdok = matrix.todok()
    matrixGraph = nx.Graph()
    matrixGraph.add_edges_from(connectome)
    # reorganize columns and rows into block - diagonal form
    components = list(nx.connected_components(matrixGraph))
    components.sort(key=len)
    size = matrix.rows
    blocks = []
    index_maps_h = {}
    index_maps_v = {}
    block_stride = 0
    for component in components:
        component = list(component)
        indices_0 = sorted(list(set([x[0] for x in component])))
        indices_1 = sorted(list(set([x[1] for x in component])))
        block_size = len(indices_0)
        mapping_v = {i0: k for k, i0 in zip(range(block_size), indices_0)}
        mapping_h = {i1: k for k, i1 in zip(range(block_size), indices_1)}
        index_maps_v.update({(k+block_stride): i0 for
                             k, i0 in zip(range(block_size), indices_0)})
        index_maps_h.update({(k+block_stride): i1 for
                             k, i1 in zip(range(block_size), indices_1)})
        block = {(mapping_v[c[0]], mapping_h[c[1]])
                  : matrixdok[(c[0], c[1])] for c in component}
        blocks.append(sp.SparseMatrix(block_size, block_size, block))
        block_stride += block_size
    # calculate the permutted bases
    col_reordering = [index_maps_h[k] for k in range(size)]
    row_reordering = [index_maps_v[k] for k in range(size)]
    return sp.Matrix(sp.BlockDiagMatrix(*blocks)), blocks, col_reordering, row_reordering

Qnums = namedtuple('Qmlms', ['ml1', 'ms1', 'ml2', 'ms2'])
LSQnums = namedtuple('QLSJM', ['L', 'S', 'J', 'MJ'])
AntiQnums = namedtuple('AntiQ', ['ls', 'ms'])

def antisymmetrize(qet):
    qet_len = len(list(qet.dict.keys())[0])
    assert qet_len % 2 == 0
    split = qet_len//2
    biz_qet = Qet(
        {Qnums(*k[split:], *k[:split]): v for k, v in qet.dict.items()})
    return sp.S(1)/sp.S(2)*(qet + (-1)*biz_qet)


def antisymmetrize_sym(qet):
    '''
    Once the Qet has been made assymetrical
    the order of ls and ms values is irrelevant.
    As such the keys of Qets can be made to be
    namedtuples whose values are frozensets.
    '''
    qet_len = len(list(qet.dict.keys())[0])
    assert qet_len % 2 == 0
    split = qet_len//2
    biz_qet = Qet(
        {Qnums(*k[split:], *k[:split]): v for k, v in qet.dict.items()})
    asym_qet = sp.S(1)/sp.S(2)*(qet - biz_qet)
    final_qet = Qet(
        {AntiQnums(tuple(sorted([k[0], k[2]])), tuple(sorted([k[1], k[3]]))): v for k, v in asym_qet.dict.items()})
    return final_qet


def LS_basis_equiv_electrons(l_orbital):
    '''
    Returns  the  LS  coupled  basis  for  a  pair of equivalent
    electrons with a shared orbital angular momentum l.
    The  resulting  Qets  are  eigenvectors of L^2, S^2, J^2 and
    M_J.

    Parameters
    ----------
    l (int): orbital quantum number

    Returns
    -------
    asym (dict) : keys are (L,S,J,M) namedtuples and values are Qets
    whose keys are (ml1, ms1, ml2, ms2) namedtuples.
    '''
    l = l_orbital
    s1, l1 = sp.S(1)/2, sp.S(round(l*2))/2
    s2, l2 = s1, l1
    ml1s, ms1s = list(range(-l1, l1+1)), list(np.arange(-s1, s1+1))
    ml2s, ms2s = ml1s, ms1s
    summands = {}
    Ls = list(np.arange(np.abs(l2-l1), l1+l2+1))
    Ss = list(np.arange(np.abs(s2-s1), s1+s2+1))
    for L in Ls:
        mLs = np.arange(-L, L+1)
        for S in Ss:
            mSs = np.arange(-S, S+1)
            Js = np.arange(np.abs(L-S), L+S+1)
            for ml1, ml2, ms1, ms2, mL, mS, J in product(ml1s, ml2s, ms1s, ms2s, mLs, mSs, Js):
                mJs = np.arange(-J, J+1, 1)
                for mJ in mJs:
                    c1 = cg.eva(L, S, J, mL, mS, mJ)
                    if c1 == 0:
                        continue
                    c2 = cg.eva(s1, s2, S, ms1, ms2, mS)
                    if c2 == 0:
                        continue
                    c3 = cg.eva(l1, l2, L, ml1, ml2, mL)
                    if c3 == 0:
                        continue
                    c = c1*c2*c3
                    if c != 0:
                        if (L, S, J, mJ) not in summands.keys():
                            summands[(L, S, J, mJ)] = []
                        ϕ = Qet({Qnums(ml1, ms1, ml2, ms2): c})
                        summands[LSQnums(L, S, J, mJ)].append(ϕ)
    totals = OrderedDict()
    for k, v in summands.items():
        tee = sum(v, Qet({}))
        totals[LSQnums(*k)] = tee
    asym = OrderedDict()
    for k, v in totals.items():
        asym_qet = antisymmetrize(v)
        if len(asym_qet.dict) != 0:
            asym[k] = asym_qet
    return asym


def Lz_total(qnums, coeff, l1, l2):
    J1 = coeff*Qet({qnums: qnums[0]})
    J2 = coeff*Qet({qnums: qnums[2]})
    return J1+J2


def Lplus_total(qnums, coeff, l1, l2):
    ml1, ms1, ml2, ms2 = qnums.ml1, qnums.ms1, qnums.ml2, qnums.ms2
    J1 = coeff * Qet({Qnums(ml1+1, ms1, ml2, ms2): sp.sqrt(l1*(l1+1)-ml1*(ml1+1)),
                     Qnums(ml1, ms1, ml2+1, ms2): sp.sqrt(l2*(l2+1)-ml2*(ml2+1))})
    return J1


def Lminus_total(qnums, coeff, l1, l2):
    ml1, ms1, ml2, ms2 = qnums
    J1 = coeff * Qet({Qnums(ml1-1, ms1, ml2, ms2): sp.sqrt(l1*(l1+1)-ml1*(ml1-1)),
                     Qnums(ml1, ms1, ml2-1, ms2): sp.sqrt(l2*(l2+1)-ml2*(ml2-1))})
    return J1


def Lx_total(qnums, coeff, l1, l2):
    return sp.S(1)/2*(Lplus_total(qnums, coeff, l1, l2) + Lminus_total(qnums, coeff, l1, l2))


def Ly_total(qnums, coeff, l1, l2):
    return -sp.I*sp.S(1)/2*(Lplus_total(qnums, coeff, l1, l2) + (-1)*Lminus_total(qnums, coeff, l1, l2))


def L_total_squared(qet, l):
    Lx1 = sum([Lx_total(k, v, l, l) for k, v in qet.dict.items()], Qet())
    Lx2 = sum([Lx_total(k, v, l, l) for k, v in Lx1.dict.items()], Qet())
    Ly1 = sum([Ly_total(k, v, l, l) for k, v in qet.dict.items()], Qet())
    Ly2 = sum([Ly_total(k, v, l, l) for k, v in Ly1.dict.items()], Qet())
    Lz1 = sum([Lz_total(k, v, l, l) for k, v in qet.dict.items()], Qet())
    Lz2 = sum([Lz_total(k, v, l, l) for k, v in Lz1.dict.items()], Qet())
    return Lx2+Ly2+Lz2


def Sz_total(qnums, coeff, l1, l2):
    J1 = coeff*Qet({qnums: qnums[1]})
    J2 = coeff*Qet({qnums: qnums[3]})
    return J1+J2


def Splus_total(qnums, coeff, l1, l2):
    ml1, ms1, ml2, ms2 = qnums.ml1, qnums.ms1, qnums.ml2, qnums.ms2
    J1 = coeff * Qet({Qnums(ml1, ms1+1, ml2, ms2): sp.sqrt(l1*(l1+1)-ms1*(ms1+1)),
                     Qnums(ml1, ms1, ml2, ms2+1): sp.sqrt(l2*(l2+1)-ms2*(ms2+1))})
    return J1


def Sminus_total(qnums, coeff, l1, l2):
    ml1, ms1, ml2, ms2 = qnums
    J1 = coeff * Qet({Qnums(ml1, ms1-1, ml2, ms2): sp.sqrt(l1*(l1+1)-ms1*(ms1-1)),
                     Qnums(ml1, ms1, ml2, ms2-1): sp.sqrt(l1*(l1+1)-ms2*(ms2-1))})
    return J1


def Sx_total(qnums, coeff, l1, l2):
    return sp.S(1)/2*(Splus_total(qnums, coeff, l1, l2)
                      + Sminus_total(qnums, coeff, l1, l2))


def Sy_total(qnums, coeff, l1, l2):
    return -sp.I*sp.S(1)/2*(Splus_total(qnums, coeff, l1, l2)
                            + (-1)*Sminus_total(qnums, coeff, l1, l2))


def S_total_squared(qet, l):
    Sx1 = sum([Sx_total(k, v, l, l) for k, v in qet.dict.items()], Qet())
    Sx2 = sum([Sx_total(k, v, l, l) for k, v in Sx1.dict.items()], Qet())
    Sy1 = sum([Sy_total(k, v, l, l) for k, v in qet.dict.items()], Qet())
    Sy2 = sum([Sy_total(k, v, l, l) for k, v in Sy1.dict.items()], Qet())
    Sz1 = sum([Sz_total(k, v, l, l) for k, v in qet.dict.items()], Qet())
    Sz2 = sum([Sz_total(k, v, l, l) for k, v in Sz1.dict.items()], Qet())
    return Sx2+Sy2+Sz2


def Jz_total(qnums, coeff, l1, l2, s1, s2):
    return Lz_total(qnums, coeff, l1, l2) + Sz_total(qnums, coeff, s1, s2)


def Jplus_total(qnums, coeff, l1, l2, s1, s2):
    return Lplus_total(qnums, coeff, l1, l2) + Splus_total(qnums, coeff, s1, s2)


def Jminus_total(qnums, coeff, l1, l2, s1, s2):
    return Lminus_total(qnums, coeff, l1, l2) + Sminus_total(qnums, coeff, s1, s2)


def Jx_total(qnums, coeff, l1, l2, s1, s2):
    return sp.S(1)/2*(Jplus_total(qnums, coeff, l1, l2, s1, s2)
                      + Jminus_total(qnums, coeff, l1, l2, s1, s2))


def Jy_total(qnums, coeff, l1, l2, s1, s2):
    return -sp.I*sp.S(1)/2*(Jplus_total(qnums, coeff, l1, l2, s1, s2)
                            + (-1)*Jminus_total(qnums, coeff, l1, l2, s1, s2))


def J_total_squared(qet, l, s):
    Sx1 = sum([Jx_total(k, v, l, l, s, s) for k, v in qet.dict.items()], Qet())
    Sx2 = sum([Jx_total(k, v, l, l, s, s) for k, v in Sx1.dict.items()], Qet())
    Sy1 = sum([Jy_total(k, v, l, l, s, s) for k, v in qet.dict.items()], Qet())
    Sy2 = sum([Jy_total(k, v, l, l, s, s) for k, v in Sy1.dict.items()], Qet())
    Sz1 = sum([Jz_total(k, v, l, l, s, s) for k, v in qet.dict.items()], Qet())
    Sz2 = sum([Jz_total(k, v, l, l, s, s) for k, v in Sz1.dict.items()], Qet())
    return Sx2+Sy2+Sz2

def half_integer_fixer(nums):
    return [sp.S(int(num*2))/2 for num in nums]

def mrange(j):
    # returns ranges that work for half integers
    j = sp.S(int(2*j))/2
    return list(-j+i for i in range(2*j+1))

def lrange(j1,j2):
    # return the range of possible total j
    return list(abs(j1-j2) + i for i in range((j1+j2)-abs(j1-j2)+1))

class AngularMomentum():
    '''
    Angular momentum and all its wonders.
    '''
    remembered_additions = {}
    @classmethod
    def add(cls, ls):
        '''
        Given  an  iterable  of  angular  momenta  to  add  up, this
        function    determines   states   which   are   simultaneous
        eigenvectors  of  L_total^2,  and  L_total_z. In addition to
        these  two the resulting kets also preserve the "trajectory"
        taken  to arrive to them, in the sense that the intermediate
        L_total  are  also  kept: these can be used to differentiate
        the  several  ways in which a particular ket (eigenvector of
        L_total^2  and  L_total_)  can  be  arrived at. Furthermore,
        these  additional  steps    (labeled L123...m) also stand in
        for eigenvalues of L123....m_total^2.

        Parameters
        ----------
        ls  : (iterable)
            an   iterable   consisting   of  n  non-negative
            integers or half-integers.

        Returns
        -------
        {
         'kets' : (OrderedDict)
                keys  are (mL123..n, L123..n, L123...(n-1), ... L12)
                and  values  are  qets  whose  keys are (ml_1, ml_2,
                ml_3, ... , ml_n) tuples}
         'uncoupled_basis' : (list)
                values are tuples of all the (ml_1, ml_2, ..., ml_n)
                combos
        }
        '''
        assert all(map(lambda x: x >= 0, ls))
        ls = tuple(half_integer_fixer(ls))
        if len(ls) == 1:
            ls = ls[0]
            root_kets = OrderedDict([((mls,ls), Qet({(mls,):1})) \
                                         for mls in mrange(ls)])
            return {'kets': root_kets,
                    'uncoupled_basis': [(mls,) for mls in mrange(ls)]}
        uncoupled_basis = list(product(*[mrange(l) for l in ls]))
        print("Basis will include %d kets." % len(uncoupled_basis))
        if ls in cls.remembered_additions.keys():
            return cls.remembered_additions[ls]
        ls_original = tuple(ls)
        # if there are more than two to add, take the last one "l_next"
        # and collect the other ones "ls". This is done recursively
        # until "ls" contains only two values of l to add, at which
        # point the the root_kets are simply the states of the first
        # ls
        if len(ls) > 2:
            ls, l_next = ls[:-1], ls[-1]
            print("Coupling %s to %s" % (str(ls), str(l_next)))
            l_root = cls.add(ls)
        else:
            # when there's only two the root_kets are simply
            # the kets of ls[0]
            ls, l_next = ls[0], ls[1]
            print("Coupling %s to %s" % (str(ls), str(l_next)))
            root_kets = OrderedDict([((mls,ls), Qet({(mls,):1})) \
                                         for mls in mrange(ls)])
            l_root = {'kets': root_kets}
        ml_nexts = mrange(l_next)
        kets = l_root['kets']
        summands = {}
        # doing the sum over the keys of the included kets
        # simplifies an iterator that would otherwise be
        # more complex
        # to take the adequate sum, the terms that correspond
        # to each sum are collected in the keys of the dictionary
        # summands
        # they keys of kets are tuples such that the first element
        # is the value mL at that stage, L12...n is the second,
        # L12...(n-1) the third, ....
        for ketroot_nums, ketroot in kets.items():
            l_stems = lrange(ketroot_nums[1], l_next)
            for l_stem in l_stems:
                ml_stems = mrange(l_stem)
                for ml_next, ml_stem in product(ml_nexts, ml_stems):
                    c = cg.eva(ketroot_nums[1], l_next, l_stem, \
                               ketroot_nums[0], ml_next, ml_stem)
                    if c == 0:
                        continue
                    combo = (ml_stem, l_stem, *ketroot_nums[1:])
                    if combo not in summands.keys():
                        summands[combo] = []
                    summands[combo].append(ketroot * Qet({(ml_next,): c}))
        coupled_kets = OrderedDict()
        # add up all the summands enclosed in each list as
        # keyed by each tuple (ml_stem, l_stem, ...)
        for k, v in summands.items():
            coupled_kets[k] = sum(v, Qet({}))
        ordered_keys = sorted(coupled_kets.keys())
        # order the keys of the coupled_kets
        coupled_kets_ordered = OrderedDict([(k,coupled_kets[k]) \
                                            for k in ordered_keys])
        assert len(coupled_kets) == len(uncoupled_basis), \
                        '%d %d' % (len(coupled_kets), len(uncoupled_basis))
        cls.remembered_additions[ls_original] = {'kets': coupled_kets_ordered,
                            'uncoupled_basis': uncoupled_basis}
        return {'kets': coupled_kets_ordered,
                'uncoupled_basis': uncoupled_basis}
    @classmethod
    def clear_add_cache(cls):
        cls.remembered_additions = {}
