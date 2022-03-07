#!/usr/bin/env python3

######################################################################
#                  _____     The matrix  ___                         #
#                 / ____/___  ____  ____/ (_)__  _____               #
#                / / __/ __ \/ __ \/ __  / / _ \/ ___/               #
#               / /_/ / /_/ / /_/ / /_/ / /  __(__  )                #
#               \____/\____/\____/\__,_/_/\___/____/                 #
#                                                                    #
######################################################################

import networkx as nx
import sympy as sp
import numpy as np
from uncertainties import ufloat

def block_form(matrix):
    '''
    This  function  takes  a matrix and rearranges its columns and rows so
    that  it  is  in  block  diagonal form. It returns a tuple whose first
    element is the reorganized matrix, whose second element is the are the
    blocks  themselves, whose third element is the reordering of the rows,
    and whose fourth element is the reordering of the columns.

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


def eigenvalue_disambiguate(eigenvals, interpolant_order=2, runway=10):
    '''
    Useful  to  identify  "branches" in the spectrum of hermitian matrices
    M(x_i) which are functions of a single continous parameter x.
    
    This  function tracks the eigenvalues as the parameter of the matrices
    is  varied. This is done by implementing continuity constraints in the
    values and extrapolated values of the created branches.
    
    This  function  is  agnostic  to  the parameters that have defined the
    given  eigenvalues,  and  only  relies on them being evenly spaced and
    ordered.

    Parameters
    ----------

    runway  (int)  :  how  many  points are used to train the interpolants

    eigenvals  (np.array)  :  an  array  where eigenvals[i] represents the
    eigenvalues of M(x_i)

    Returns
    -------
    mbranches  (np.array):  with mbraches[j] representing the j-th identified
    branch of the given operator; mbranches has as many rows as eigenvals has
    columns, and as many columns as eigenvals had rows.

    '''
    mbranches = {idx:[energeigenvals] for idx, energeigenvals in enumerate(eigenvals[0])}
    for idx, col in enumerate(eigenvals[1:]):
        supplied_branches = []
        for energeigenvals in col:
            interpolants = {}
            poleigenvalsvalues = {}
            if idx < runway:
                dists = []
                for branchidx in mbranches:
                    last_energeigenvals = mbranches[branchidx][-1]
                    dists.append((branchidx,np.abs(last_energeigenvals-energeigenvals), energeigenvals))
                dists = list(sorted(dists, key = lambda x: x[1]))
                while True:
                    belonging = dists[0][0]
                    if belonging not in supplied_branches:
                        supplied_branches.append(belonging)
                        mbranches[belonging].append(dists[0][2])
                        break
                    else:
                        dists = dists[1:]
            else:
                # for each branch build an interpolation and extrapolate +1
                extrapols = []
                for branchidx in mbranches:
                    branch_energies = mbranches[branchidx][-runway:]
                    x = range(runway)
                    if branchidx in interpolants:
                        poleigenvalsvalue = poleigenvalsvalues[branchidx]
                        print('.')
                        1/0
                    else:
                        poleigenvalsfit = np.poly1d(np.polyfit(x,branch_energies, interpolant_order))
                        interpolants[branchidx] = poleigenvalsfit
                        poleigenvalsvalue = poleigenvalsfit(runway)
                        poleigenvalsvalues[branchidx] = poleigenvalsvalue
                    extrapols.append((branchidx, np.abs(poleigenvalsvalue-energeigenvals)))
                extrapols = list(sorted(extrapols, key = lambda x: x[1]))
                while True:
                    belonging = extrapols[0][0]
                    if belonging not in supplied_branches:
                        supplied_branches.append(belonging)
                        mbranches[belonging].append(energeigenvals)
                        break
                    else:
                        extrapols = extrapols[1:]
    return np.array(list(mbranches.values()))

def degen_remove(array, utol = 1e-3):
    '''
    This  function  takes  an  array,  and  returns  the  columns that are
    different to one another to within the provided tolerance.

    Parameters
    ----------
    array  (np.array): Where array[i] are interpreted as the columns to be
    compared.

    Returns
    -------
    uniqrows  (np.array):  an array whose rows are all different to within
    the provided tolerance.

    '''
    uniqrows = [array[0]]
    tol = utol * len(uniqrows[0])
    for idx in range(1,len(array)):
        branch = array[idx]
        diffs = [np.sqrt(np.sum(np.abs(branch-ubranch)**2)) > tol for ubranch in uniqrows]
        if all(diffs):
            uniqrows.append(branch)
    return np.array(uniqrows)

def vector_upgrade(vector, pivots, fullbasis):
    '''
    Give a vector whose components are initially only in a subspace
    of an original vector space, promote it by adding zeros in  the
    adequate places.

    Parameters
    ----------
    vector (np.array): the vector in a subspace of the larger vector
    space.

    pivots (list): contains the indices in the larger vector  space
    that are represented in the coefficients of the given vector.
    
    fullbasis (list or np.array): a list of ordered unit vectors.

    Example
    -------

    >> vector = [1,2,3]
    >> pivots = [0,2,4]
    >> fullbasis = np.eye(7)
    >> vector_upgrade(vector, pivots, fullbasis)

    array([1,0,2,0,3,0,0])

    Added on Jan-17 2022-01-17 11:43:49
    '''
    return np.sum(np.array([coeff*fullbasis[pivot] 
                for coeff, pivot in zip(vector, pivots)]), axis=0)

def block_diagonalize(blocks, notches, symbols_rep, final_sort = True, assume_hermitian = True):
    '''
    If a matrix has been put into block diagonal form, this function
    can help in finding its spectrum by patching together the eigen-
    values and eigenvectors of the blocks.

    Parameters
    ----------

    blocks  (list): iterable  whose  elements are symbolic sp.Matrix
    notches (list): a list of integers which represent the subspaces
                to which the given  blocks belong to in the original
                ordering of the matrix
    symbols_rep (dict): a dictionary with substitutions that convert
                         all the given matrices into numerical ones.
    
    Returns
    -------

    (all_eigenvals, all_eigenvects)   (tuple)
    all_eigenvals (np.array): an array with all the blocks eigenvalues
    all_eigenvects (np.array):  an  array  whose  j-th  column  is  an 
                                   eigenvector of the j-th eigenvalue.
    
    Added on Jan-17 2022-01-17 11:44:29
    '''
    dims = list(map(lambda x: x.rows, blocks))
    total_dim = sum(dims)
    fullbasis = np.eye(total_dim)
    all_eigenvals = np.zeros((total_dim))
    all_eigenvects =  np.zeros((total_dim, total_dim))
    cursor = 0
    counter = 0
    for block, offsets in zip(blocks, notches):
        block_dim = block.rows
        block = np.array(block.subs(symbols_rep)).astype(np.complex64)
        eigenvals, eigenvects = np.linalg.eigh(block)
        if assume_hermitian:
            eigenvals = np.real(eigenvals)
        eigenvects = eigenvects.T
        all_eigenvals[cursor:cursor+block_dim] = eigenvals
        offsets = notches[cursor:cursor+block_dim]
        for eigenvect in eigenvects:
            all_eigenvects[counter] = vector_upgrade(eigenvect, offsets, fullbasis)
            counter += 1
        cursor += block_dim
    if final_sort:
        sorter = np.argsort(all_eigenvals)
        all_eigenvals, all_eigenvects = all_eigenvals[sorter], all_eigenvects[sorter]
    all_eigenvects = all_eigenvects.T
    return all_eigenvals, all_eigenvects

def uncertain_eigenvalsh(smatrix, vars, num_trials):
    '''
    Given  a  hermitian  matrix  (sp.Matrix)  that  depends  on  a  set of
    variables,  and  given  the best estimates and uncertainties for their
    values,  this  function  determines  the  eigenvalues  of  the  matrix
    together with its estimated uncertainty.

    It  assumes that the errors in the parameters are normaly distributed,
    such  that  the  uncertainties  given  in  its  parameters  equal  the
    standard deviation for the corresponding normal distributions.

    No checks are made that the matrix be hermitian.

    Parameters
    ----------
    smatrix  (sp.Matrix):  a matrix with free_symbols equal to the keys of
    vars.

    num_trials  (int):  how  many samples will be taken in the Monte Carlo
    evaluation of the eigenvalues.

    vars  (OrderedDict):  keys  being  the varibles in smatrix, and values
    being equal to ufloat.

    Returns
    -------
    uncertain_eigenvalues  (np.array):  an  np.array  whose  elements  are
    ufloat,  which  correspond  to  the  eigenvalues  of the given matrix,
    ordered from smallest to largest.
    '''
    lambda_matrix = sp.lambdify(list(vars.keys()), smatrix)
    trial_matrices = []
    for _ in range(num_trials):
            exp_params = [np.random.normal(varvalue.nominal_value,
                                          varvalue.std_dev) 
                                            for varvalue in vars.values()]
            this_ham = lambda_matrix(*exp_params)
            trial_matrices.append(np.array(this_ham,dtype=np.complex64))
    trial_matrices = np.array(trial_matrices)
    eigenvals = np.sort(np.linalg.eigvalsh(trial_matrices))
    best_estimates = np.mean(eigenvals, axis=0)
    stds = np.std(eigenvals, axis=0)
    uncertain_eigenvalues = np.array([ufloat(b,s) for b,s in zip(best_estimates, stds)])
    return uncertain_eigenvalues