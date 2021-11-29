#!/usr/bin/env python

######################################################################
#                               ___                                  #
#                              |__ \ ___                             #
#                              __/ // _ \                            #
#                             / __//  __/                            #
#                            /____/\___/                             #
#                                                                    #
######################################################################

# This  script  generates  the  wave  functions  and the electrostatic
# interaction  matrices  for  all  crystallographic  point groups, the
# matrices  are  given  as  a function of the smallest set of possible
# exchange and Coulomb integrals.
#
# A  LaTeX  output  is  produced  in the folder latex_output_dir and a
# pickle  is  saved with the result of the calculations for the matrix
# elements of the electrostatic interaction.
#
# In  addition  to  this,  a LaTeX output for the notation used in the
# determinantal states is also produced.
#
# Do not run this script with python but with ipython.

import sympy as sp
import numpy as np
import time
from qdef import *
from misc import *
from itertools import product
from IPython.display import HTML, display, Math, Latex
from collections import OrderedDict
from joblib import Parallel, delayed
import multiprocessing
import os, sys
from sympy.physics.wigner import clebsch_gordan as ClebschG

num_cores = multiprocessing.cpu_count()

# reload data on terms and labels for components
twoe_terms = pickle.load(open('./data/2e-terms.pkl','rb'))
new_labels = pickle.load(open('./data/components_rosetta.pkl','rb'))

latex_output_dir = '/Users/juan/Library/Mobile Documents/com~apple~CloudDocs/iCloudFiles/Theoretical Division/'
latex_output_fname = os.path.join(latex_output_dir,'termwaves.tex')

progressprint = print

if not os.path.exists(latex_output_dir):
    print("Inexistent output directory for LaTeX output, please edit.")
    sys.exit()

def overlinesqueegee(s):
    '''
    Going back from the overline shorthand for spin down,
    acting on a single set of quantum numbers.
    '''
    if 'overline' in str(s):
        spin = -sp.S(1)/2
        comp = sp.Symbol(str(s).replace('\\overline{','')[:-1])
    else:
        spin = sp.S(1)/2
        comp = s
    return (comp, spin)

def spin_restoration(qet):
    '''
    Going back from the overline shorthand for spin down,
    acting on a qet.
    '''
    new_dict = {}
    for k,v in qet.dict.items():
        k = (*overlinesqueegee(k[0]),*overlinesqueegee(k[1]))
        new_dict[k] = v
    return Qet(new_dict)

def tuplerecovery(ft):
    '''the inverse of composite_symbol'''
    return tuple(map(sp.Symbol,sp.latex(ft)[1:-1].split(',')))

def simplify_qet(qet):
    '''
    simplify all the values of the given qet
    '''
    new_dict = {k:sp.simplify(v) for k,v in qet.dict.items()}
    return Qet(new_dict)

def composite_symbol(x):
    '''
    quickly make a four element symbol
    '''
    return sp.Symbol('(%s)'%(','.join(list(map(sp.latex,x)))))

def coulomb_energy_matrices(group_label, verbose = False):
    '''
    version: 1637708720
    '''
    start_time = time.time()
    group = CPGs.get_group_by_label(group_label)
    component_labels = {k:list(v.values()) for k,v in new_labels[group_label].items()}
    group.component_labels = component_labels
    group.irrep_arrays = {ir: {gk: np.array(im) for gk, im in irms.items()} for ir, irms in group.irrep_matrices.items()}
    terms = twoe_terms[group_label]
    components = group.component_labels
    ir_arrays = group.irrep_arrays
    generators = group.generators
    # if verbose:
    #     progress = display('',display_id=True)
    if verbose:
        msg = group_label + " Computing the symbolic expression for the configuration matrices ..."
        progressprint(msg)
    configs = {}
    config_supplement = {}
    for term_key, term in terms.items():
        for state_key, state in zip(term.state_keys, term.states):
            (Γ1, Γ2, Γ3, γ3, S, mSz) = state_key
            α = sp.Symbol(sp.latex(Γ1).lower())*sp.Symbol(sp.latex(Γ2).lower())
            if α not in configs.keys():
                configs[α] = {}
                config_supplement[α] = {}
            if (S,Γ3) not in configs[α].keys():
                configs[α][(S,Γ3)] = []
                config_supplement[α][(S,Γ3)] = []
            configs[α][(S,Γ3)].append(state)
            config_supplement[α][(S,Γ3)].append(state_key)

    # within each configuration
    config_matrices = {}
    for config in configs.keys():
        # display(config)
        # within each term
        term_matrices = {}
        for term_key, states in configs[config].items():
            # find the brackets between all possible pairs of states
            ham_matrix = []
            for state0 in states:
                ham_row = []
                state0 = spin_restoration(state0)
                for state1 in states:
                    # recover the spin for the determinantal states
                    pstate = state1
                    state1 = spin_restoration(state1)
                    # determine the det braket between these two states (an operator in between is assumed and omitted)
                    detbraket = state0.dual()*state1 # changehere
                    # simplify this braket of detstates into brakets between regular states
                    regbraket = Qet({})
                    for k,v in detbraket.dict.items():
                        γ1, m1, γ2, m2, γ1p, m1p, γ2p, m2p = k
                        # k0 = (γ1, m1, γ2, m2, γ1p, m1p, γ2p, m2p)
                        # k1 = (γ1, m1, γ2, m2, γ2p, m2p, γ1p, m1p)
                        k0 = (γ1, γ2, γ1p, γ2p)
                        k1 = (γ1, γ2, γ2p, γ1p)
                        # enforce orthogonality wrt. spin
                        if (m1 == m1p) and (m2 == m2p):
                            qplus = Qet({k0:v})
                        else:
                            qplus = Qet({})
                        if (m1 == m2p) and (m2 == m1p):
                            qminus = Qet({k1:-v})
                        else:
                            qminus = Qet({})
                        regbraket += (qplus+qminus)
                    ham_row.append(regbraket)
                ham_matrix.append(ham_row)
            term_matrices[term_key] = ham_matrix
        config_matrices[config] = term_matrices

    if verbose:
        msg = group_label + " Computing non-redundant quadruples of irrep symbols ..."
        progressprint(msg)

    combos_4 = set()
    for ir1, ir2, ir3, ir4 in product(*([group.irrep_labels]*4)):
        altorder1 = (ir3, ir2, ir1, ir4)
        altorder2 = (ir1, ir4, ir3, ir2)
        altorder3 = (ir3, ir4, ir1, ir2)
        if (altorder1 in combos_4) or\
           (altorder2 in combos_4) or\
           (altorder3 in combos_4):
            continue
        combos_4.add((ir1, ir2, ir3, ir4))

    # determine four symbol identities
    # keep only the non-trivial ones
    # and determine which ones are identically zero

    def four_symb_ids(ircombo):
        ircombo = tuple(ircombo)
        ir1, ir2, ir3, ir4 = ircombo
        comp_to_idx = [{c: idx for idx, c in enumerate(component_labels[ir])} for ir in ircombo]
        comp_idx = [list(range(group.irrep_dims[ir])) for ir in ircombo]
        a_idx0, a_idx1, a_idx2, a_idx3,  = np.indices([group.irrep_dims[ir] for ir in ircombo])
        component_tensor = [components[ir] for ir in ircombo]
        integral_identity_sector = []
        zeros = []
        def idx_to_component_tuple(idx_quple):
            idx_quple = tuple(idx_quple)
            return tuple([component_tensor[i][idx_quple[i]] for i in range(4)])
        for R in generators:
            Rtensor = np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                    np.conjugate(ir_arrays[ir1][R]),
                                    np.conjugate(ir_arrays[ir2][R]), 0),
                                    ir_arrays[ir3][R], 0),
                                    ir_arrays[ir4][R], 0)
            for idx0, idx1, idx2, idx3 in product(*comp_idx):
                γcombo = idx_to_component_tuple((idx0,idx1,idx2,idx3))
                Rt = Rtensor[idx0,:,idx1,:,idx2,:,idx3,:]
                nonZee = (Rt != 0) # non-zero
                nonZ = np.array([a_idx0[nonZee], a_idx1[nonZee], a_idx2[nonZee], a_idx3[nonZee]]).T
                deect = {(idx_to_component_tuple(r)):Rt[tuple(r)] for r in nonZ}
                qet = Qet({γcombo:1}) - Qet(deect)
                # remove trivial identities
                if len(qet.dict) > 0:
                    integral_identity_sector.append(qet)
                if len(qet.dict) == 1:
                    zeros.append(γcombo)
        return (ircombo, integral_identity_sector), (ircombo, zeros)

    if verbose:
        msg = group_label + " Computing 4-symbol identities ..."
        progressprint(msg)

    if group_label in ['T_{h}', 'O_{h}', 'D_{6h}']:
        out = (Parallel(n_jobs = 1)(delayed(four_symb_ids)(combo) for combo in combos_4))
        integral_identities = dict(list(map(lambda x: x[0], out)))
        all_zeros = dict(list(map(lambda x: x[1], out)))
    else:
        out = (Parallel(n_jobs = num_cores)(delayed(four_symb_ids)(combo) for combo in combos_4))
        integral_identities = dict(list(map(lambda x: x[0], out)))
        all_zeros = dict(list(map(lambda x: x[1], out)))

    # use the zeros to simplify integral_identities
    great_identities = {irc:[] for irc in integral_identities}
    counter = 0
    for ircombo in integral_identities:
        zeros = all_zeros[ircombo]
        for identity in integral_identities[ircombo]:
            new_qet = Qet({})
            for k,v in identity.dict.items():
                if k in zeros:
                    counter += 1
                    continue
                else:
                    new_qet+= Qet({k: v})
            if len(new_qet.dict) == 0:
                continue
            great_identities[ircombo].append(new_qet)

    #     print("%d terms zeroed out" % counter)
    #     final_constraints = sum(list(map(len,great_identities.values())))
    #     initial_constraints = sum(list(map(len,integral_identities.values())))
    #     print("%d initial constraints, %d final constraints" % (initial_constraints, final_constraints))

    if verbose:
        msg = group_label + " Assuming real basis functions and computing corresponding identities ..."
        progressprint(msg)
    real_var_simplifiers = {irc:[] for irc in great_identities}
    kprimes = set()
    # this has to run over all the quadruples of irs
    for ir1, ir2, ir3, ir4 in product(*([group.irrep_labels]*4)):
        real_var_simplifier = {}
        ircombo = (ir1, ir2, ir3, ir4)
        components = [component_labels[ir] for ir in ircombo]
        for γ1, γ2, γ3, γ4 in product(*components):
            k = (γ1, γ2, γ3, γ4)
            kalt1 = (γ3, γ2, γ1, γ4)
            kalt2 = (γ1, γ4, γ3, γ2)
            kalt3 = (γ3, γ4, γ1, γ2)
            if kalt1 in kprimes:
                real_var_simplifier[k] = kalt1
            elif kalt2 in kprimes:
                real_var_simplifier[k] = kalt2
            elif kalt3 in kprimes:
                real_var_simplifier[k] = kalt3
            else:
                real_var_simplifier[k] = k
                kprimes.add(k)
        real_var_simplifiers[ircombo] = real_var_simplifier

    real_var_full_simplifier = {}
    for ircombo in real_var_simplifiers:
        real_var_full_simplifier.update(real_var_simplifiers[ircombo])

    if verbose:
        msg = group_label + " Simplifying systems of equations ..."
        progressprint(msg)

    def identity_solver(ircombo, these_ids, these_zeros):
        problem_vars = list(set(sum([list(identity.dict.keys()) for identity in these_ids],[])))
        big_ma = np.array([awe.vec_in_basis(problem_vars) for awe in these_ids])
        zero_addendum = {k:{} for k in these_zeros}
        big_mat = sp.Matrix(big_ma)
        big_mat = sp.re(big_mat) + sp.I*sp.im(big_mat)
        rref_mat, pivots = big_mat.rref()
        num_rows = rref_mat.rows
        num_cols = rref_mat.cols
        rref_ma_non_zero = [rref_mat[row,:] for row in range(num_rows) if (sum(np.array(rref_mat[row,:])[0] == 0) != num_cols)]
        rref_mat_non_zero = sp.Matrix(rref_ma_non_zero)
        varvec = sp.Matrix(list(map(composite_symbol,problem_vars)))
        eqns = rref_mat_non_zero*varvec
        ssol = sp.solve(list(eqns), dict=True)
        assert len(ssol) in [0,1]
        if len(ssol) == 0:
            sol_dict = {}
        else:
            sol_dict = ssol[0]
        sol_dict = {tuplerecovery(k):{tuplerecovery(s):v.coeff(s) for s in v.free_symbols} for k,v in sol_dict.items()}
        sol_dict.update(zero_addendum)
        return (ircombo, sol_dict)
    all_sols = dict(Parallel(n_jobs = num_cores)(delayed(identity_solver)(ircombo, great_identities[ircombo], all_zeros[ircombo]) for ircombo in great_identities))

    super_solution = {}
    for ircombo in all_sols:
        super_solution.update(all_sols[ircombo])

    def simplifier(qet):
        simp_ket = Qet({})
        for k,v in qet.dict.items():
            simp_ket += Qet({real_var_full_simplifier[k]:v})
        true_qet = Qet({})
        for k,v in simp_ket.dict.items():
            if k in super_solution:
                true_qet += v*Qet(super_solution[k])
            else:
                true_qet += Qet({k:v})
        return true_qet

    def simplify_config_matrix(ir1ir2, S, ir3):
        config_matrix = config_matrices[ir1ir2][(S,ir3)]
        num_rows = len(config_matrix)
        simple_matrix = [[simplifier(config_matrix[row][col]) for col in range(num_rows)] for row in range(num_rows)]
        return simple_matrix

    if verbose:
        msg = group_label + " Simplifying configuration matrices ..."
        progressprint(msg)

    simple_config_matrices = {k:{} for k in config_matrices}
    for ir1ir2 in config_matrices.keys():
        for term_key in config_matrices[ir1ir2]:
            S, Γ3 = term_key
            simple_config_matrices[ir1ir2][term_key] = simplify_config_matrix(ir1ir2, S, Γ3)

    twotuplerecovery = tuplerecovery
    if verbose:
        msg = group_label + " Creating all 2-symbol identities ..."
        progressprint(msg)

    # this   integral_identities  dictionary  will  have  as  keys
    # 2-tuples  of  irreps  and  its  values  will  be lists whose
    # elements  are  2-tuples whose first elements are 2-tuples of
    # irrep  components  and  whose values are qets whose keys are
    # 2-tuples  of  irrep components and whose values are numeric.
    # These 2-tuples represent a braket with the an invariant operator
    # operator in between.

    integral_identities_2 = {}
    ir_mats = group.irrep_matrices
    for ir1, ir2 in product(*([group.irrep_labels]*2)):
        # To simplify calculations this part
        # can only be done over quadruples in a standard
        # order.
        # Whatever is missed here is then brough back in
        # by the reality relations.
        altorder1 = (ir2, ir1, ir1, ir4)
        if (altorder1 in integral_identities_2):
            continue
        integral_identity_sector = []
        components = [component_labels[ir] for ir in [ir1, ir2]]
        comp_to_idx = [{c: idx for idx, c in enumerate(component_labels[ir])} for ir in [ir1, ir2]]
        for R in group.generators:
            R_id = {}
            for γ1, γ2 in product(*components):
                for γ1p, γ2p in product(*components):
                        val = sp.conjugate(ir_mats[ir1][R][comp_to_idx[0][γ1],comp_to_idx[0][γ1p]]) *\
                              ir_mats[ir2][R][comp_to_idx[1][γ2],comp_to_idx[1][γ2p]]
                        if val== 0:
                            continue
                        key = (γ1, γ2)
                        if key not in R_id.keys():
                            R_id[key] = []
                        R_id[key].append( Qet({(γ1p,γ2p): val}) )
            R_id_total = [(key, sum(R_id[key], Qet({}))) for key in R_id.keys()]
            R_id_total = [q for q in R_id_total if len(q[1].dict)>0]
            integral_identity_sector.extend(R_id_total)
        integral_identities_2[(ir1,ir2)] = integral_identity_sector

    # For solving the linear system it is convenient
    # to have everything on one side of the equation.
    if verbose:
        msg = group_label + " Creating set of 2 symbol identities ..."
        progressprint(msg)
    identities_2 = {}

    for ircombo in integral_identities_2:
        these_ids = []
        for v in integral_identities_2[ircombo]:
            lhs, rhs = v
            diff = Qet({lhs:1}) - rhs
            if len(diff.dict) > 0:
                these_ids.append(diff)
        identities_2[ircombo] = these_ids

    # If an equation has only one term, then
    # that immediately means that that term is zero.
    if verbose:
        msg = group_label + " Finding trivial zeros ..."
        progressprint(msg)

    # first determine which ones have to be zero
    better_identities_2 = {irc:[] for irc in identities_2}
    all_zeros_2 = {}
    for ircombo in identities_2:
        zeros = []
        for identity in identities_2[ircombo]:
            if len(identity.dict) == 1:
                zeros.append((list(identity.dict.keys())[0]))
            else:
                better_identities_2[ircombo].append(identity)
        all_zeros_2[ircombo] = zeros

    # use them to simplify things.
    if verbose:
        msg = group_label + " Using them to simplify things ..."
        progressprint(msg)

    great_identities_2 = {irc:[] for irc in identities_2}
    for ircombo in better_identities_2:
        for identity in better_identities_2[ircombo]:
            new_qet = Qet({})
            for k,v in identity.dict.items():
                if k in all_zeros_2[ircombo]:
                    continue
                else:
                    new_qet+= Qet({k: v})
            if len(new_qet.dict) == 0:
                continue
            great_identities_2[ircombo].append(new_qet)

    # Inside of a two-symbol braket one may do three exchanges
    # that must result in the same value. That if the wave
    # functions are assumed to be real-valued.
    if verbose:
        msg = group_label + " Creating reality identities ..."
        progressprint(msg)
    real_var_simplifiers_2 = {irc:[] for irc in great_identities_2}
    # this has to run over all the quadruples of irs
    kprimes = set()
    for ir1, ir2 in product(*([group.irrep_labels]*2)):
        real_var_simplifier = {}
        ircombo = (ir1, ir2)
        components = [component_labels[ir] for ir in ircombo]
        for γ1, γ2 in product(*components):
            k = (γ1, γ2)
            kalt = (γ2, γ1)
            # If for a given key I find that its
            # switched version has already been seen
            # Then that key has to be mapped to be
            # mapped to the key already present.
            if kalt in kprimes:
                real_var_simplifier[k] = kalt
            else:
                real_var_simplifier[k] = k
                kprimes.add(k)
        real_var_simplifiers_2[ircombo] = real_var_simplifier

    # For each 2-tuple of irreps
    # create a system of symbolic solutions
    # and let sympy solve that.
    # For each 2-tuple of irreps
    # the end result is a dictionary
    # whose keys represent the dependent
    # brakets and whose values are the
    # relation that those dependent values
    # have with the independent brakets.
    # As such, when these dictionaries are
    # used on an expression, everything should
    # then be given in terms of indepedent brakets.

    if verbose:
        msg = group_label + " Solving for independent 2-symbol brakets ..."
        progressprint(msg)
    all_sols_2 = {irc:[] for irc in great_identities_2}

    for ircombo in great_identities_2:
        zeros = all_zeros_2[ircombo]
        problem_vars = list(set(sum([list(identity.dict.keys()) for identity in great_identities_2[ircombo]],[])))
        big_ma = [awe.vec_in_basis(problem_vars) for awe in great_identities_2[ircombo]]
        big_mat = sp.Matrix(big_ma)
        big_mat = sp.re(big_mat)+sp.I*sp.im(big_mat)
        rref_mat, pivots = big_mat.rref()
        num_rows = rref_mat.rows
        num_cols = rref_mat.cols
        rref_ma_non_zero = [rref_mat[row,:] for row in range(num_rows) if (sum(np.array(rref_mat[row,:])[0] == 0) != num_cols)]
        rref_mat_non_zero = sp.Matrix(rref_ma_non_zero)
        varvec = sp.Matrix(list(map(composite_symbol,problem_vars)))
        eqns = rref_mat_non_zero*varvec
        ssol = sp.solve(list(eqns), dict=True)
        all_sols[ircombo] = ssol
        assert len(ssol) in [0,1]
        if len(ssol) == 0:
            sol_dict = {}
        else:
            sol_dict = ssol[0]
        zero_addendum = {k:{} for k in all_zeros_2[ircombo]}
        sol_dict = {twotuplerecovery(k):{twotuplerecovery(s):v.coeff(s) for s in v.free_symbols} for k,v in sol_dict.items()}
        sol_dict.update(zero_addendum)
        all_sols_2[ircombo] = sol_dict

    # This final dictionary is unnecessary but
    # simplifies calling the replacements onto
    # a symbolic expression.
    if verbose:
        msg = group_label + " Creating a dictionary with all the 2-symbol replacements ..."
        progressprint(msg)

    super_solution_2 = {}
    for ircombo in all_sols_2:
        super_solution_2.update(all_sols_2[ircombo])

    # for a given electron config
    more_ids = {e_config:[] for e_config in simple_config_matrices}
    for e_config in simple_config_matrices:
        for term in simple_config_matrices[e_config]:
            the_matrix = simple_config_matrices[e_config][term]
            num_rows = len(the_matrix)
            num_cols = len(the_matrix)
            the_state_keys = config_supplement[e_config][term]
            the_key_matrix = {}
            for row_idx in range(num_rows):
                Γ3_row = the_state_keys[row_idx][2]
                γ3_row = the_state_keys[row_idx][3]
                for col_idx in range(num_cols):
                    Γ3_col = the_state_keys[col_idx][2]
                    γ3_col = the_state_keys[col_idx][3]
                    matrix_val = the_matrix[row_idx][col_idx]
                    the_key_matrix[(γ3_row,γ3_col)] = matrix_val
            # now go over the keys of super_solution_2
            # and if one of those keys matches with a key in the_key_matrix
            # do something about it
            for k in super_solution_2:
                if k in the_key_matrix:
                    v = super_solution_2[k]
                    # this matrix element
                    matrix_element = the_key_matrix[k]
                    # must be identified with the sum
                    # as given in v
                    matrix_equiv = sum([kv*the_key_matrix[km] for km, kv in v.items()],Qet({}))
                    identity = (matrix_element - matrix_equiv) #=0
                    if len(identity.dict) != 0:
                        more_ids[e_config].append(identity)
                        problem_vars = list(set(sum([list(identity.dict.keys()) for identity in more_ids[e_config]],[])))
    if verbose:
        msg = group_label + " Simplifying numeric values of qets ..."
        progressprint(msg)

    more_ids = {e_config:list(filter(lambda x: len(x.dict) > 0,list(map(simplify_qet,more_ids[e_config])))) for e_config in more_ids}

    if verbose:
        msg = group_label + " Solving for dependent vars in terms of independent ones ..."
        progressprint(msg)

    all_sols_2_4 = {irc:[] for irc in more_ids}

    for e_config in more_ids:
        problem_vars = list(set(sum([list(identity.dict.keys()) for identity in more_ids[e_config]],[])))
        big_ma = [awe.vec_in_basis(problem_vars) for awe in more_ids[e_config]]
        big_mat = sp.Matrix(big_ma)
        big_mat = sp.re(big_mat)+sp.I*sp.im(big_mat)
        rref_mat, pivots = big_mat.rref()
        num_rows = rref_mat.rows
        num_cols = rref_mat.cols
        rref_ma_non_zero = [rref_mat[row,:] for row in range(num_rows) if (sum(np.array(rref_mat[row,:])[0] == 0) != num_cols)]
        rref_mat_non_zero = sp.Matrix(rref_ma_non_zero)
        varvec = sp.Matrix(list(map(composite_symbol, problem_vars)))
        eqns = rref_mat_non_zero*varvec
        ssol = sp.solve(list(eqns), dict=True)
        all_sols_2_4[e_config] = ssol
        assert len(ssol) in [0,1]
        if len(ssol) == 0:
            sol_dict = {}
        else:
            sol_dict = ssol[0]
        sol_dict = {twotuplerecovery(k):{twotuplerecovery(s):v.coeff(s) for s in v.free_symbols} for k,v in sol_dict.items()}
        all_sols_2_4[e_config] = sol_dict

    # flatten into a single dictionary of replacements
    fab_solution_2_4 = {}
    for e_config in all_sols_2_4:
        fab_solution_2_4.update(all_sols_2_4[e_config])
    if verbose:
        msg = group_label + (" There are %d less independent variables ..." % len(fab_solution_2_4))
        progressprint(msg)

    def simplifier_f(qet):
        true_qet = Qet({})
        for k,v in qet.dict.items():
            if k in fab_solution_2_4:
                true_qet += v*Qet(fab_solution_2_4[k])
            else:
                true_qet += Qet({k:v})
        return true_qet

    def simplify_config_matrix_f(ir1ir2, S, ir3):
        config_matrix = simple_config_matrices[ir1ir2][(S,ir3)]
        num_rows = len(config_matrix)
        simple_matrix = [[simplifier_f(config_matrix[row][col]) for col in range(num_rows)] for row in range(num_rows)]
        #simple_matrix = sp.Matrix(simple_matrix)
        return simple_matrix

    if verbose:
        msg = group_label + " Making final simplifications ..."
        progressprint(msg)

    final_config_matrices = {k:{} for k in config_matrices}
    for ir1ir2 in config_matrices.keys():
        for term_key in config_matrices[ir1ir2]:
            S, Γ3 = term_key
            final_config_matrices[ir1ir2][term_key] = simplify_config_matrix_f(ir1ir2, S, Γ3)

    time_taken = time.time() - start_time
    if verbose:
        msg = group_label + (" Finished in %.1f s." % time_taken)
        progressprint(msg)

    return final_config_matrices

def simplify_qet_values(qet):
    sqetdict = {k:sp.simplify(v) for k,v in qet.dict.items()}
    return Qet(sqetdict)

def det_simplify(qet):
    '''
    Simplification from antisymmetry of composing elements.
    '''
    qet_dict = qet.dict
    best_qet = {}
    for key, coeff in qet_dict.items():
        ikey = (*key[3:],*key[:3])
        current_keys = list(best_qet.keys())
        if ikey in current_keys:
            best_qet[ikey] += -coeff
            continue
        if key not in current_keys:
            best_qet[key] = coeff
        else:
            best_qet[key] += coeff
    return Qet(best_qet)

def as_determinantal_ket(qet):
    qdict = {}
    for k,v in qet.dict.items():
        if k[1] >= 0:
            k0 = sp.Symbol(str(k[0]))
        else:
            k0 = sp.Symbol('\\overline{%s}'%str(k[0]))
        if k[3] >= 0:
            k1 = sp.Symbol(str(k[2]))
        else:
            k1 = sp.Symbol('\\overline{%s}'%str(k[2]))
        qdict[(k0,k1)] = v
    qet = Qet(qdict)
    qet = qet*(1/qet.norm())
    ket = qet.as_ket(fold_keys=True)
    ket = sp.latex(ket).replace('\\right\\rangle','\\right|')
    return sp.Symbol(ket), qet

# with new labels and checking for num of waves
def format_empheq(qet):
    chunk_size = 3
    qet_parts = list(qet.items())
    qets = [Qet(dict(qp[i:(i+chunk_size)])) for i in range(0,len(qet_parts),chunk_size)]

def num_waves(ir0,ir1):
    if ir0 == ir1:
        return sp.binomial(group.irrep_dims[ir0]*2, 2)
    else:
        return group.irrep_dims[ir0]*group.irrep_dims[ir1]*4

def full_waves():
    done = []
    size = 0
    for k,v in group.product_table.odict.items():
        if (k[1],k[0]) in done:
            continue
        if k[1] == k[0]:
            size += sp.binomial(group.irrep_dims[k[1]]*2,2)
        else:
            size += sum(list(map(lambda x: group.irrep_dims[x]*4, v)))
        done.append(k)
    return size

def format_empheq(qet):
    if len(qet.dict) == 0:
        return r'''
\vspace{0.2cm}
\boxed{\Delta{E}=0}
\vspace{0.2cm}'''
    else:
        chunk_size = 3
        qet_parts = list(qet.dict.items())
        qets = [Qet(dict(qet_parts[i:(i+chunk_size)])) for i in range(0,len(qet_parts),chunk_size)]
        chunks = [sp.latex(qet.as_braket()).replace('|','||').replace(r'\right.',r'\right.\!\!') for qet in qets]
        chunks[0] = r'\Delta{E}='+chunks[0]
        for chunk_idx in range(1,len(chunks)):
            if chunks[chunk_idx][0] != '-':
                chunks[chunk_idx] = '+' + chunks[chunk_idx]
        lqets = '\\\\\n'.join(chunks)
        nice_output = r'''\vspace{-0.5cm}
\begin{empheq}[box=\fbox]{gather*}
%s
\end{empheq}''' % lqets
        return nice_output

if __name__=='__main__':
    all_matrices = {}
    for group_label in CPGs.all_group_labels:
        all_matrices[group_label] = coulomb_energy_matrices(group_label, True)
    
    term_energies_for_printout = {}
    for group_label, term_energies in all_matrices.items():
        for e_config, terms  in term_energies.items():
            for term_key, term_matrix in terms.items():
                term_energy = term_matrix[0][0]
                term_energy = simplify_qet_values(term_energy)
                term_energies_for_printout[(group_label, e_config, term_key)] = term_energy
    
    pickle.dump(term_energies_for_printout, open('./Data/term_energies_for_printout.pkl','wb'))
    
    all_final_outputs = []
    tally_waves = {}
    single_col_groups = ['C_{1}','C_{2}']
    num_multicols = {group_label:2 for group_label in CPGs.all_group_labels}
    for scg in single_col_groups:
        num_multicols[scg] = 1
    archival_terms = {}
    # progress = display('',display_id=True)
    for group_counter, group_label in enumerate(CPGs.all_group_labels):
        msg = "Working on group %s ..." % group_label
        progressprint(msg)
        group = CPGs.get_group_by_label(group_label)
        s1, s2 = sp.S(1)/2, sp.S(1)/2
        Ss  = [0,1]
        m1s = [-sp.S(1)/2, sp.S(1)/2]
        m2s = [-sp.S(1)/2, sp.S(1)/2]
        group_CGs = group.CG_coefficients
        flat_labels = dict(sum([list(l.items()) for l in list(new_labels[group_label].values())],[]))
        group_CGs = {(flat_labels[k[0]], flat_labels[k[1]], flat_labels[k[2]]):v for k,v in group_CGs.items()}
        summands = OrderedDict()
        group.new_component_labels = OrderedDict([(ir, list(new_labels[group_label][ir].values())) for ir in group.irrep_labels])
        for Γ1, Γ2, Γ3, m1, m2, S in product(group.irrep_labels,
                                         group.irrep_labels,
                                         group.irrep_labels,
                                         m1s,
                                         m2s,
                                         Ss):
            for γ1, γ2, γ3 in product(group.new_component_labels[Γ1],
                                      group.new_component_labels[Γ2],
                                      group.new_component_labels[Γ3]):
                for mSz in range(S,-S-1,-1):
                    sCG = ClebschG(s1, s2, S, m1, m2, mSz)
                    if (γ1, γ2, γ3) not in group_CGs.keys():
                        continue
                    else:
                        gCG = group_CGs[(γ1, γ2, γ3)]
                    coeff = sCG*gCG
                    key = (Γ1, Γ2, Γ3, γ3, S, mSz)
                    if (Γ2,m2,γ2) == (Γ1,m1,γ1):
                        continue
                    if coeff!=0:
                        if key not in summands.keys():
                            summands[key] = []
                        summands[key].append(Qet({(Γ1,γ1,m1,Γ2,γ2,m2):coeff}))
        total_qets = OrderedDict([(k, sum(v,Qet({}))) for k,v in summands.items()])
        best_qets = OrderedDict([(k, det_simplify(v)) for k,v in total_qets.items()])
        best_qets = OrderedDict([(k, v) for k,v in best_qets.items() if len(v.dict)!=0])
        terms = OrderedDict()
        done_keys = []
        for k, v in best_qets.items():
            (Γ1, Γ2, Γ3, γ3, S, mSz) = k
            equiv_key = (Γ2, Γ1, Γ3, γ3, S, mSz)
            term_pair = (Γ3, S)
            if term_pair not in terms.keys():
                terms[term_pair] = OrderedDict()
            if equiv_key in done_keys:
                continue
            terms[term_pair][k] = v
            done_keys.append(k)
        final_terms = OrderedDict()
        for term in terms:
            ir, S = term
            states = terms[term]
            one_term = Term({'irrep': ir, 'S': S, 'states': states})
            final_terms[term] = one_term
    #     print("Manufacturing latex output ....")
        max_counter = 3
        if group_counter == 0:
            print_outs = ['\n\\newpage\n\n\\section{Terms and wave functions}\n\\subsection{Group $%s$}\n\n\\begin{center} \\underline{Component labels} \n\\vspace{0.2cm}\n' % group_label]
        else:
            print_outs = ['\n\\doublerulefill\n\\subsection{Group $%s$} \n\n\\begin{center}\n\n\\underline{Component labels} \n\\vspace{0.2cm}\n' % group_label]
        for running_idx, irrep_symbol in enumerate(group.irrep_labels):
            components_for_printing = ','.join(list(map(sp.latex, group.new_component_labels[irrep_symbol])))
            if running_idx < (len(group.irrep_labels)-1):
                print_outs.append('$%s:\\{%s\\}$ || ' % (sp.latex(irrep_symbol), components_for_printing))
            else:
                print_outs.append('$%s:\\{%s\\}$' % (sp.latex(irrep_symbol), components_for_printing))
        if num_multicols[group_label] == 1:
            print_outs.append('\\end{center}\n\n')
        else:
            print_outs.append('\\end{center}\n\n\\begin{multicols}{%d}\n\n' % num_multicols[group_label])
        supreme_states = {}
        total_waves = 0
        end_terms = {}
        for one_term_k, one_term in final_terms.items():
            term_symb = r'{{}}^{{{M}}}\!{ir}'.format(M=(2*one_term_k[1]+1), ir = sp.latex(one_term_k[0]))
            term_symb = sp.Symbol(term_symb)
            print_outs.append('\n\\hrulefill\n\\subsubsection{$%s$}\n\\vspace{0.25cm}\n \\begin{center} \n' % sp.latex(term_symb))
            counter = 0
            prev_α = ''
            end_terms[one_term_k] = []
            origins = []
            for state_key, state in one_term.states.items():
                (Γ1, Γ2, Γ3, γ3, S, mSz) = state_key
                αl = sp.Symbol(sp.latex(Γ1).lower())*sp.Symbol(sp.latex(Γ2).lower())
                α = Γ1 * Γ2
                multiplicity = 2*S+1
                term_symb = r'{{}}^{{{M}}}\!{ir}'.format(M=(2*S+1), ir = sp.latex(Γ3))
                state_symbol = '\\Psi_{%d}(%s,%s,M\!=\!%d,%s)' % (counter+1, sp.latex(α).lower(), term_symb, mSz, sp.latex(γ3))
                state_symbol = sp.Symbol(state_symbol)
                v_simple = Qet({(k[1],k[2],k[4],k[5]):v for k,v in state.dict.items()})
                v_det, q_qet = as_determinantal_ket(v_simple)
                end_terms[one_term_k].append(q_qet)
                origins.append(state_key)
                sup_key = α
                if sup_key not in supreme_states.keys():
                    supreme_states[sup_key] = []
                supreme_states[sup_key].append(v_det)
                pout = "$\\textcolor{blue}{%s} = %s$\\vspace{0.1cm}\n" % (sp.latex(state_symbol), sp.latex(v_det))
                pout = sp.Symbol(pout)
                if prev_α != α:
                    if prev_α == '':
                        print_outs.append('\n\n $\\textcolor{red}{%s}$ \n\n' % (sp.latex(α).lower()))
                    else:
                        print_outs.append('\n\\vspace{0.25cm}\n\n $\\textcolor{red}{%s}$ \n\n' % (sp.latex(α).lower()))
                    term_energy = term_energies_for_printout[(group_label, αl, one_term_k[-1::-1])]
                    print_outs.append('\n\n %s \n\n' % (format_empheq(term_energy)))
                prev_α = α
                print_outs.append(sp.latex(pout))
                counter += 1
                total_waves += 1
                done_keys.append(state_key)
            end_terms[one_term_k] = Term({'irrep': one_term_k[0], 'S': one_term_k[1], 'states': end_terms[one_term_k], 'state_keys': origins})
            print_outs.append('\n\\end{center}\n')
        archival_terms[group_label] = end_terms
        if num_multicols[group_label] != 1:
            print_outs.append('\n\\end{multicols}\n')
        irrep_combos = list(combinations_with_replacement(group.irrep_labels,2))
        total_waves_groundtruth = sum([num_waves(ir0,ir1) for ir0, ir1 in irrep_combos])
        tally_waves[group_label] = (total_waves,total_waves_groundtruth,full_waves())
        all_wavefunctions = '\n'.join(print_outs)
        all_wavefunctions = all_wavefunctions.replace("^'","^{'}").replace("^''","^{''}")
        all_final_outputs.append(all_wavefunctions)
    print("Saving LaTeX output to file ...")
    super_final_output = '\n'.join(all_final_outputs)
    open(latex_output_fname,'w').write(super_final_output)
    print("Saving to pickle ...")
    pickle.dump(archival_terms, open('./Data/2e-terms.pkl','wb'))
    
    # with new labels
    group_chunks = []
    for group_label in CPGs.all_group_labels:
        group = CPGs.get_group_by_label(group_label)
        all_spin_combos = OrderedDict()
        spin_states = [sp.Symbol(r'\alpha'), sp.Symbol(r'\beta')]
        group.new_component_labels = {ir:list(new_labels[group_label][ir].values()) for ir in group.irrep_labels}
        for irrep in group.irrep_labels:
            spin_orbitals = []
            for component0, component1 in product(group.new_component_labels[irrep],group.new_component_labels[irrep]):
                for s0, s1 in product(spin_states,spin_states):
                    spin_orbital = (component0, component1, s0, s1)
                    if (component1, component0, s1, s0) not in spin_orbitals:
                        if (component1, s1) != (component0, s0):
                            spin_orbitals.append(spin_orbital)
            spin_orbital_symbols = {}
            for spin_orbital in spin_orbitals:
                component0, component1, s0, s1 = spin_orbital
                if s0 == spin_states[0]:
                    cstar0 = sp.latex(component0)
                else:
                    cstar0 = r'\overline{%s}' % sp.latex(component0)
                if s1 == spin_states[0]:
                    cstar1 = sp.latex(component1)
                else:
                    cstar1 = r'\overline{%s}' % sp.latex(component1)
                spin_orbital_symbols[spin_orbital] = sp.Symbol(r'$|%s%s|$' % (cstar0, cstar1))
            all_slater_symbols = []
            for s in list(spin_orbital_symbols.values()):
                all_slater_symbols.append(sp.latex(s))
            all_slater_symbols = ', '.join(all_slater_symbols)
            all_slater_symbols = '\\begin{center}\n%s\n\\end{center}' % all_slater_symbols
            all_slater_symbols = ('\\begin{center}\n $%s{\\cdot}%s:$ \n\\end{center}\n\n' % (sp.latex(irrep),sp.latex(irrep))) + all_slater_symbols
            all_spin_combos[irrep] = all_slater_symbols
        final_output = '\n\n'.join(all_spin_combos.values())
        final_output = ('\\hrulefill \n \\begin{center} Group $%s$ \\end{center} \n' % group_label) + final_output
        group_chunks.append(final_output)
    super_output = '\n'.join(group_chunks)
    super_output = super_output.replace("^'","^{'}").replace("^''","^{''}")
    open(os.path.join(latex_output_dir,'spin-orbitals.tex'),'w').write(super_output)


    