#!/usr/bin/env python3

######################################################################
#                  _____     matrix        ___                       #
#                 / ____/___  ____  ____/ (_)__  _____               #
#                / / __/ __ \/ __ \/ __  / / _ \/ ___/               #
#               / /_/ / /_/ / /_/ / /_/ / /  __(__  )                #
#               \____/\____/\____/\__,_/_/\___/____/                 #
#                                                                    #
######################################################################

import networkx as nx
import sympy as sp
import numpy as np

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
