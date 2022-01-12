#!/usr/bin/env python3

import sympy as sp
from qdef import *
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import warnings
import pickle
import sys
from datetime import datetime

save_to_pickle = True
pickle_fname = './data/symmetry_bases_standard.pkl'
computed_groups = CPGs.all_group_labels
in_parallel = False

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
        all_comparisons[R] = (check, direct_way, irrep_way)
        all_checks[R] = check
    if sum(all_checks.values()) == len(all_checks):
        all_good = True
    else:
        all_good = False
    if full_output:
        return all_comparisons
    else:
        return all_good

def symmetry_adapted_basis_standard(group_label, lmax, verbose=False):
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

    Parameters
    ----------
    group_label    (str): a label for a crystallographic point group
    lmax           (int): up  to  which  value  of the l the bases will be 
                          constructed.

    Returns
    -------

    all_symmetry_bases (dict): a dictionary whose keys correspond to group
    labels,  whose  values  are  dictionaries,  whose  keys  correspond to
    symbols    for   irreducible   representations,   whose   values   are
    dictionaries,  whose  keys  are  values  of l, whose values are lists,
    whose   elements   lists,  whose  elements  are  qets  that  represent
    combinations  of spherical harmonics. How many qets there are on these
    final   lists  corresponds  to  the  dimension  of  the  corresponding
    irreducible  representation,  and  how  many of these lists there are,
    corresponds to a possible degeneracy in the correspondig value of l.

    '''
    # The GramSchmidt routine from sympy fails in an odd case,
    # because of this I had to replace it with a custom version.
    GramSchmidtFun = GramSchmidtAlt
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
            # display(bigmatrix)
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
                chunks = [tuple(map(tuple,chunk)) for chunk in chunks if (sp.Matrix(chunk).rank() == irrep_dim)]
                chunks = list(set(chunks))

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
                normalized = GramSchmidtFun(flat_rows, orthonormal=True)
                parts = []
                for deg in range(degeneracy):
                    chunk = list(map(list,normalized[deg*irrep_dim : deg*irrep_dim + irrep_dim]))
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

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("This script requires one argument, which equals l_max.")
        sys.exit()
    lmax = int(sys.argv[1])
    symmetry_adapted_basis = symmetry_adapted_basis_standard
    if in_parallel:
        sym_bases = Parallel(n_jobs = num_cores)(delayed(symmetry_adapted_basis)(group_label, lmax, True) for group_label in computed_groups)
        all_the_symmetry_bases = dict(zip(computed_groups, sym_bases))
    else:
        all_the_symmetry_bases = {}
        for group_label in computed_groups:
            all_the_symmetry_bases[group_label] = symmetry_adapted_basis(group_label, lmax, True)
    total_checks = {}
    checklist = [basis_check(group_label, irrep_symb, basis) \
                for group_label in CPGs.all_group_labels \
                for irrep_symb in CPGs.get_group_by_label(group_label).irrep_labels \
                for l in range(lmax) \
                for (basis_idx, basis) in enumerate(all_the_symmetry_bases[group_label][irrep_symb][l])
                ]
    all_checks = (len(checklist) == sum(checklist))
    if all_checks:
        print("Out of %d instances all %d passed the test." % (len(checklist), sum(checklist)))
        if save_to_pickle:
            print("Saving to pickle...")
            pickle.dump(all_the_symmetry_bases, open(pickle_fname,'wb'))
    else:
        print("Out of %d instances only %d passed the test." % (len(checklist), sum(checklist)))
