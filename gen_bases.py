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
pickle_fname = './data/symmetry_bases_sc_and_standard_12.pkl'
computed_groups = CPGs.all_group_labels
# computed_groups = ['D_{3}']
in_parallel = False

def qetrealsimpler(qets):
    '''
    Given  a  list of qets, return a set that spans the same subspace, but
    with real components.

    Parameters
    ----------
    qets  (list) with qdefcore.Qet values

    Returns
    -------
    (2-tuple)  :  (list), (str) If the string is "real" then a solution to
    make  for  a  real combination was succesful and the returned list has
    the  resultant  qets,  if  the  string  is  "non-real" then a solution
    couldn't be found and the qets returned are the original ones.
    '''
    # determine the common basis
    print('.',end='')
    common_basis = list(set(sum([list(q.dict.keys()) for q in qets],[])))
    # find the coefficients in that basis
    vs = [q.vec_in_basis(common_basis) for q in qets]
    realities = [all([sp.im(c) == 0 for c in vec]) for vec in vs]
    if all(realities):
        return qets, "real"
    A = sp.Matrix(vs)
    # A = sp.re(A) + sp.I*sp.im(A)
    Ap = A.pinv()
    ApA = sp.simplify(Ap*A)
    # eigensys = ApA.eigenvects()
    # nicevects = [[(ve.T) for ve in v[2]] for v in eigensys if v[0]==1][0]
    ApAm = ApA - sp.eye(ApA.rows)
    eigensys = ApAm.nullspace()
    nicevects = [(ve.T) for ve in eigensys]
    # nicevects = [list(v/v.norm()) for v in nicevects]
    realities = [all([sp.im(c) == 0 for c in nicevect]) for nicevect in nicevects]
    irealities = [all([sp.re(c) == 0 for c in nicevect]) for nicevect in nicevects]
    bettervects = []
    for nicevect, reality, ireality in zip(nicevects, realities, irealities):
        if reality:
            bettervects.append(list(nicevect))
            continue
        if ireality:
            bettervects.append([list(sp.I*v) for v in nicevect])
            continue
    if len(vs) == len(bettervects):
        betterqets = [Qet({cb:v for cb, v in zip(common_basis, bettervect)}) for bettervect in bettervects]
        betterqets = sorted(betterqets, key= lambda x: len(x.dict))
        return betterqets, "real"
    else:
        return qets, "non real"

def irrep_check(group, a_basis_list, irrep, l):
    '''
    This  function  checks  if  a  list  of qets (which are interpreted as
    superpositions  of Ylms for fixed l), are a basis for the given irrep.
    This  is done by calculating the matrices that represent the effect of
    applying  all  of  the  group  operations, and then checking to see if
    those matrices are unitary, satisfy the multiplication table, and have
    the corresponding character for the given irrep.

    This function could easily be modified to return the matrices that the
    given basis induces.

    Parameters
    ----------

    group    (qdef.CrystalGroup)

    a_basis_list   (list[list[qdef.Qet]]): interpreted  as  superpositions 
        of standard spherical harmonics. Each list must have as many qets
        as the dimension of the corresponding irrep. Several groups may be
        given to account for possible degeneracy of the bases.

    irrep   (sp.Symbol): symbol for one of the group irreps.

    l      (int): the  value  of  l  from which the given qets are seen to
        originate from.
    
    Returns
    -------
    all_checks (list): a list of nested lists, of which there are as many
    as groups of qets were given. The enclosed lists have three Bool, the
    first one of which is the check for the multiplication table, the 2nd
    of which is the of characters of the induced matrices, and the  third
    being the check against the matrices being unitary.
    '''
    all_checks = []
    for a_basis in a_basis_list:
        lbasis = [(l,m) for m in range(-l,l+1)]
        basis_matrix = sp.Matrix([b.vec_in_basis(lbasis) for b in a_basis]).T
        basis_matrix = Dagger(basis_matrix)
        induced_matrices = {}
        char_checks = {}
        unitarity_checks = {}
        for group_op, op_params in group.euler_angles.items():
            D_matrix = (op_params[3]**l)*sp.Matrix([[Wigner_D(l,m,n,*op_params[:3]) for n in range(-l,l+1)] for m in range(-l,l+1)]).T
            R_rows = []
            for basis_element in a_basis:
                vector = sp.Matrix(basis_element.vec_in_basis(lbasis))
                t_vector = D_matrix*vector
                row = (basis_matrix*t_vector).T
                R_rows.append(row)
            induced_matrices[group_op] = sp.Matrix(R_rows).T
            induced_matrices[group_op] = sp.re(induced_matrices[group_op]) + sp.I * sp.im(induced_matrices[group_op])
            diff = sp.N(sp.simplify(induced_matrices[group_op]).trace() - sp.simplify(group.irrep_matrices[irrep][group_op]).trace(), chop=True)
            char_check = (diff == 0)
            if not char_check:
                warnings.warn("Character mismatch.")
            char_checks[group_op] = char_check
            ru = sp.simplify(induced_matrices[group_op] * Dagger(induced_matrices[group_op]))
            ru = sp.re(ru) + sp.I*sp.im(ru)
            unitarity_checks[group_op] =  (ru == sp.eye(group.irrep_dims[irrep]) )
        # check to see if the product table is satisfied
        ptable = group.multiplication_table_dict
        checks = {}
        for group_op_0 in group.group_operations:
            for group_op_1 in group.group_operations:
                group_op_01 = ptable[(group_op_0,group_op_1)]
                target = induced_matrices[group_op_01]
                directp = sp.simplify(induced_matrices[group_op_0]*induced_matrices[group_op_1])
                directp = sp.re(directp) + sp.I * sp.im(directp)
                target = sp.re(target) + sp.I * sp.im(target)
                pcheck = sp.N(directp - target, chop=True) == sp.zeros(target.rows)
                if not pcheck:
                    warnings.warn("Multiplication table mismatch.")
                checks[(group_op_0,group_op_1)] = pcheck
        all_checks.append((all(checks.values()), all(char_checks.values()), all(unitarity_checks.values())))
    return all_checks

def symmetry_adapted_basis_v_real(group_label, lmax, verbose=False):
    '''
    Starting from a set of functions it may be possible to construct a new
    set which transforms according to the irreducible representations of a
    given  group.  This  is  done by using the proyection operators of the
    group.  Given  that  these  operators  may  produce linearly dependent
    combinations  some  amount of linear algebra is necessary to determine
    the largest set of linearly independent sets of functions.

    A  necessary  component  of  the  algorithm  is  knowledge  of how the
    starting  functions  transform  under  the group's operations. This is
    well  known  for  the  spherical  harmonics  in  terms of the Widger D
    matrices.

    It  may  also  be  desirable to have the resulting set of functions be
    real,  in  which case it is more convenient to start with the sine and
    cosine combinations of spherical harmonics. In this function this path
    is taken.

    After  a  set  has  been  found,  another  set,  which would transform
    accordingly  to an equivalent irreducible representation, may be found
    by applying a unitary transformation.  This  is the effect of applying
    the function real_or_imagined_global_unitary somewhere near the end of
    this function.

    Parameters
    ----------
    group_label    (str): a label for a crystallographic point group
    lmax           (int): up  to  which  value  of the l the bases will be 
                          constructed.

    Returns
    -------
    (2-tuple)  symmetry_basis_sc,  symmetry_basis: where symmetry_basis_sc
    is  the  resultant  basis  in  terms  of  sine  and  cosine  spherical
    harmonics,  and  symmetry_basis  is  for  the  result  translated into
    standard spherical harmonics.
    Both  these are nested dictionaries where the first key corresponds to
    labels  (sp.Symbol)  of  the irreducible representations of the group,
    whose  second  key  corresponds  to a value of l, and whose value is a
    list  of  lists where each list if of length equal to the dimension of
    the  irreducible  representation  (there  may   be  more than one list
    because  at  times  there's  more  than one possible basis that may be
    formed).
    '''
    GramSchmidtFun = GramSchmidtAlt
    group = CPGs.get_group_by_label(group_label)
    group_irreps = group.irrep_labels
    symmetry_basis_sc = {}
    symmetry_basis = {}
    if verbose:
        print("\n")
        print("*"*33)
        print(group_label)
        print("*"*33)
    for group_irrep in group_irreps:
        if verbose:
            print(str(group_irrep))
        # if group_irrep != sp.Symbol('E'):
        #     continue
        irrep_dim = group.irrep_dims[group_irrep]
        symmetry_basis_sc[group_irrep] = {}
        symmetry_basis[group_irrep] = {}
        irrep_matrices = group.irrep_matrices[group_irrep]
    
        for l in range(lmax+1):
            # if l!= 9:
            #     continue
            if verbose:
                print('l=',l)
                print(datetime.now())
            full_basis = [(l,m) for m in range(-l,l+1)]
            full_basis_real = [(l,m,'c') for m in range(0,l+1)] + [(l,m,'s') for m in range(1,l+1)]
            realharm_qet_basis = [Qet({(l,m,'c'):1}) for m in range(0,l+1)] + [Qet({(l,m,'s'):1}) for m in range(1,l+1)]
            standard_qet_basis = [Qet({(l,m):1}) for m in range(-l,l+1)]

            # We need two change of coordinates matrices that allow going
            # back and forth from the Y_lm^{cs} to the Y_lm.
            
            combos = [Qet({
                (l,abs(m),'c'):sp.S(1)/sp.sqrt(2),
                (l,abs(m),'s'):-sp.I/sp.sqrt(2)}).vec_in_basis(full_basis_real) for m in range(-l,0)]
            combos.append(Qet({(l,0,'c'):1}).vec_in_basis(full_basis_real))
            combos = combos + [Qet({
                (l,m,'c'): sp.S(1)/sp.sqrt(2),
                (l,m,'s'): sp.I*sp.S(1)/sp.sqrt(2)}).vec_in_basis(full_basis_real) for m in range(1,l+1)]

            # When this matrix multiplies a vector of coefficients of regular spherical harmonics
            # the results are the coefficients in the basis composed of Y_lm^{sc}

            change_of_basis_matrix = sp.Matrix(combos).T # this matrix changes from the regular Ylm to Ylm^cs

            combos2 = [Qet({(l,0):1}).vec_in_basis(full_basis)]
            combos2 = combos2 + [Qet({
                (l, m): sp.S(1)/sp.sqrt(2),
                (l,-m): 1/sp.sqrt(2)}).vec_in_basis(full_basis) for m in range(1,l+1)]
            combos2 = combos2 + [Qet({
                (l,m) : -sp.I/sp.sqrt(2),
                (l,-m): sp.I/sp.sqrt(2)}).vec_in_basis(full_basis) for m in range(1,l+1)]

            # When this matrix multiplies a vector of coefficients of Y_lm^{sc}
            # the results are the coefficients in the basis composed of standard spherical harmonics.

            change_of_basis_matrix_from_real_to_standard = sp.Matrix(combos2).T
            
            # This loop applied the proyection operators and creates a dictionary
            # all_phis whose key is the value of m that relates to it
            # and whose valuees are dictionaries whose keys are 3-tuples (t,s,'s' or 'c')

            all_phis = {}
            for m in range(0,l+1):
                phis = {}
                # for a fixed row t,
                for t in range(irrep_dim):
                    # collect of of the sums by adding over columns
                    cphi = Qet({})
                    sphi = Qet({})
                    for s in range(irrep_dim):
                        for R, DR in irrep_matrices.items():
                            dr = sp.conjugate(DR[t,s])
                            op_params = group.euler_angles[R]
                            if dr != 0:
                                if m == 0:
                                    cphi = cphi + dr*RYlm(l,m,op_params)
                                else:
                                    cphi = cphi + (1/sp.sqrt(2)) * dr    * RYlm(l,m,op_params)
                                    cphi = cphi + (1/sp.sqrt(2)) * dr    * RYlm(l,-m,op_params)
                                    sphi = sphi + (sp.I/sp.sqrt(2)) * dr * RYlm(l,m,op_params)
                                    sphi = sphi - (sp.I/sp.sqrt(2)) * dr * RYlm(l,-m,op_params)

                        coord_vec = sp.Matrix(cphi.vec_in_basis(full_basis))
                        transformed_vec = list(change_of_basis_matrix*coord_vec)
                        transformed_qet = sum([v*q for v,q in zip(transformed_vec, realharm_qet_basis) if v !=0], Qet({}))
                        phis[(t,s,'c')] = (sp.S(irrep_dim)/group.order)*transformed_qet
                        if m!=0:
                            coord_vec_sin = sp.Matrix(sphi.vec_in_basis(full_basis))
                            transformed_vec = list(change_of_basis_matrix*coord_vec_sin)
                            transformed_qet = sum([v*q for v,q in zip(transformed_vec, realharm_qet_basis) if v !=0],Qet({}))
                            phis[(t,s,'s')] = (sp.S(irrep_dim)/group.order)*transformed_qet
                all_phis[m] = phis

            # Take the qets and find the coefficients in the basis full_basis
            # this is necessary to evaluate linear independence, and useful
            # for applying the Gram-Schmidt orthonormalization process.
            coord_vecs = []
            for m,s,t in product(range(0,l+1),range(irrep_dim),range(irrep_dim)):
                coord_vecs.append(all_phis[m][(t,s,'c')].vec_in_basis(full_basis_real))
            # This second loop seems as it could be integrated in the one above
            # but this is not so, the order in which rows are added matters.
            for m,s,t in product(range(0,l+1),range(irrep_dim),range(irrep_dim)):
                if m != 0:
                    coord_vecs.append(all_phis[m][(t,s,'s')].vec_in_basis(full_basis_real))
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
                chunks = [tuple(map(tuple,chunk)) for chunk in chunks if (sp.Matrix(chunk).rank() == irrep_dim)]
                chunks = list(set(chunks))

                print("num chunks =",len(chunks),"num cycles =", cycles)
                num_combos = sp.binomial(len(chunks), cycles)
                print("Searching %d combinations." % num_combos)

                # from collections import deque
                # good_to_go = False
                # while not good_to_go:
                #     bits = [chunks[0]]
                #     good_to_go = False
                #     cursor = 1
                #     while not good_to_go:
                #         mbits = list(map(lambda x: sp.Matrix(x), bits))
                #         tot = sp.Matrix(sp.BlockMatrix(mbits))
                #         current_rank = tot.rank()
                #         if current_rank == num_lin_indep_rows:
                #             good_to_go = True
                #             break
                #         for extra_bit in chunks[cursor:]:
                #             new_bits = bits + [extra_bit]
                #             mbits = list(map(lambda x: sp.Matrix(x), new_bits))
                #             tot = sp.Matrix(sp.BlockMatrix(mbits))
                #             new_rank = tot.rank()
                #             if new_rank > current_rank:
                #                 break
                #         print(new_rank, num_lin_indep_rows)
                #         bits = new_bits
                #         cursor += 1
                #         if new_rank == num_lin_indep_rows:
                #             good_to_go = True
                #             break
                #         if cursor == len(chunks):
                #             break
                #     chunks = deque(chunks)
                #     chunks.rotate(1)
                #     chunks = list(chunks)
                # assert good_to_go == True
                counter = 0
                for bits in combinations(chunks,cycles):
                    # print('.',end='')
                    # sys.stdout.flush()
                    mbits = list(map(lambda x: [sp.Matrix(x)], bits))
                    tot = sp.Matrix(sp.BlockMatrix(mbits))
                    # numtot = sp.N(tot)
                    the_rank = tot.rank()
                    print(the_rank, num_lin_indep_rows, '%d/%d' % (counter+1, num_combos))
                    counter += 1
                    if the_rank == num_lin_indep_rows:
                        # a satisfactory subset has been found, exit
                        break
                else:
                    raise Exception("Couldn't find an adequate subset of rows.")
                # for bits in combinations(chunks,cycles):
                #     print('.',end='')
                #     sys.stdout.flush()
                #     mbits = list(map(lambda x: [sp.Matrix(x)], bits))
                #     tot = sp.Matrix(sp.BlockMatrix(mbits))
                #     # numtot = sp.N(tot)
                #     the_rank = tot.rank()
                #     # print(the_rank, num_lin_indep_rows)
                #     if the_rank == num_lin_indep_rows:
                #         # a satisfactory subset has been found, exit
                #         break
                # else:
                #     raise Exception("Couldn't find an adequate subset of rows.")
                for bit in bits:
                    good_rows.append(bit)
                if verbose:
                    print("\nOrthonormalizing ...")
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
                all_regular_qets = []
                # convert the coefficient vectors back to qets
                skip_check = False
                all_phases = []
                for part in parts:
                    normal_qets = [Qet({k: v for k,v in zip(full_basis_real,part[i]) if v!=0}) for i in range(len(part))]            
                    # print(normal_qets)
                    normal_qets = qetrealsimpler(normal_qets)[0]
                    normal_qets = [(sp.S(1)/nq.norm()) * nq for nq in normal_qets]
                    spherical_qets = []
                    for nq in normal_qets:
                        coord_vec = sp.Matrix(nq.vec_in_basis(full_basis_real))
                        transformed_vec = list(change_of_basis_matrix_from_real_to_standard*coord_vec)
                        transformed_qet = sum([v*q for v,q in zip(transformed_vec, standard_qet_basis) if v !=0],Qet({}))
                        spherical_qets.append(transformed_qet)
                    all_normal_qets.append(normal_qets)
                    phased, flag, phases = real_or_imagined_global_unitary(spherical_qets)
                    all_phases.append(phases)
                    if flag == 0:
                        skip_check = True
                    all_regular_qets.append(phased)
                if not skip_check:
                    num_different_phase_shift = len(set(tuple(map(tuple, all_phases))))
                    if num_different_phase_shift != 1:
                        warnings.warn("Phases between degenerate bases are incompatible.")
                if verbose:
                    print("Finished!")
                symmetry_basis_sc[group_irrep][l] = all_normal_qets
                symmetry_basis[group_irrep][l] = all_regular_qets
            else:
                symmetry_basis_sc[group_irrep][l] = []
                symmetry_basis[group_irrep][l] = []
    return symmetry_basis_sc, symmetry_basis

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("This script requires one argument, which equals l_max.")
        sys.exit()
    lmax = int(sys.argv[1])
    symmetry_adapted_basis = symmetry_adapted_basis_v_real
    if in_parallel:
        sym_bases = Parallel(n_jobs = num_cores)(delayed(symmetry_adapted_basis)(group_label, lmax, True) for group_label in computed_groups)
        all_the_symmetry_bases = dict(zip(computed_groups, sym_bases))
    else:
        all_the_symmetry_bases = {}
        for group_label in computed_groups:
            all_the_symmetry_bases[group_label] = symmetry_adapted_basis(group_label, lmax, True)
    total_checks = {}
    all_checks = []
    print("Checking soundness of produced bases...")
    for group_label, basis_dict in all_the_symmetry_bases.items():
        print(group_label)
        group = CPGs.get_group_by_label(group_label)
        total_checks[group_label] = {}
        for irrep_symbol, l_dict in basis_dict[1].items():
            total_checks[group_label][irrep_symbol] = []
            for l, qet_groups in l_dict.items():
                checking = sum(irrep_check(group, qet_groups, irrep_symbol, l),())
                total_checks[group_label][irrep_symbol].append(checking)
                all_checks.append(all(checking))
    # print(total_checks)
    # print(all_checks)
    if (all(all_checks)):
        print("All bases check out.")
        if save_to_pickle:
            print("Saving to pickle...")
            pickle.dump(all_the_symmetry_bases, open(pickle_fname,'wb'))
