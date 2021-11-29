#!/usr/bin/env python

######################################################################
#        _    __          ______           __                        #
#       | |  / /__  ___  / ____/___ ______/ /_____  _______  __      #
#       | | / / _ \/ _ \/ /_  / __ `/ ___/ __/ __ \/ ___/ / / /      #
#       | |/ /  __/  __/ __/ / /_/ / /__/ /_/ /_/ / /  / /_/ /       #
#       |___/\___/\___/_/    \__,_/\___/\__/\____/_/   \__, /        #
#                                                     /____/         #
######################################################################

# Nov-29 2021-11-29 11:07:56

# Given a set of Clebsch-Gordan coefficients which are not necessarily
# even  or  odd this script tries to find an adequate change of phases
# so  that  they'll  have  one  of the two symmetries. This is done by
# first  finding  the  triples  that  don't  conform  to  any specific
# symmetry,  and  then  trying  out  for each of those triples all the
# possible  phase  changes  until  one symmetric solution is found, or
# until all options are exhausted.

# This  is  successful for 20 of the 32 crystallographic point groups,
# using the CG coefficients provided by GTPack.

from qdef import *
import sympy as sp
from itertools import product
import os

pickle_name = os.path.join(module_dir, 'data','better_Vees.pkl')
rosseta_fname = os.path.join(module_dir,'data','components_rosetta.pkl')
vee_syms_pickle = os.path.join(module_dir,'data','Vee_symmetries.pkl')
new_labels = pickle.load(open(rosseta_fname,'rb'))
save_to_pickle = True

def refine_CGs(never_surrender=False, verbose=True):
    better_Vs = OrderedDict()
    success = OrderedDict()
    initial_bad_ones = OrderedDict()
    final_bad_ones = OrderedDict()
    vee_symmetries = {}
    for group_label in CPGs.all_group_labels:
        print('+'*20)
        print("Working on", group_label)
        vee_symmetries[group_label] = {'even':[], 'odd':[], 'neither':[], 'singular':[]}
        group = CPGs.get_group_by_label(group_label)
        # Grab the new labels
        component_labels = {k:list(v.values()) for k,v in new_labels[group_label].items()}
        group.component_labels = component_labels
        # Rename CGs according to new component labels
        group_CGs = group.CG_coefficients_partitioned
        flat_labels = dict(sum([list(l.items()) for l in list(new_labels[group_label].values())],[]))
        for irrep_pair, CGs in group_CGs.items():
            new_CGs = {(flat_labels[k[0]],flat_labels[k[1]],flat_labels[k[2]]):v for k,v in CGs.items()}
            group_CGs[irrep_pair] = dict(new_CGs)
        group.CG_coefficients_partitioned = dict(group_CGs)
        if verbose: print("Initializing all V coefficients to zero ...")
        V_coeff_original = {}
        for irrep_symbols in product(*[group.irrep_labels]*3):
            component_symbols = [group.component_labels[ir] for ir in irrep_symbols]
            for comp_symbs in product(*component_symbols):
                V_coeff_original[tuple(irrep_symbols)+tuple(comp_symbs)] = 0
        if verbose:
            print("Using current Clebsh-Gordan coefficients as a first ansatz to all Vees ...")
        for k, v in group.CG_coefficients_partitioned.items():
            # they outer keys correspond to the first two slots
            for cg_k, cg_v in v.items():
                # the inner keys correspond to the last three slots
                # the third slot needs to be determined as the irrep to 
                # which the last component belongs to
                the_third = [ir_label for ir_label in group.irrep_labels if cg_k[-1] in group.component_labels[ir_label]]
                assert len(the_third) == 1 # there can be only one
                the_third = the_third[0]
                V_arg = tuple(tuple(k) + (the_third,) + tuple(cg_k))
                V_coeff_original[V_arg] = sp.S(1)/sp.sqrt(group.irrep_dims[V_arg[2]])*cg_v

        Vg = V_coefficients.V_fixer(dict(V_coeff_original), {}, group, verbose=verbose)
        combo_types = Vg['combo_types']
        all_permutation_comparisons = Vg['all_permutation_comparisons']
        even_ones = [k for k,v in combo_types.items() if v == 'even']
        singular_ones = [k for k,v in combo_types.items() if v == 'singular']
        odd_ones = [k for k,v in combo_types.items() if v == 'odd']
        faulty_ones = [k for k,v in combo_types.items() if v == 'neither']
        initial_bad_ones[group_label] = len(faulty_ones)

        if verbose:
            print("\n EVEN:")
            display(sp.Matrix(list(map(lambda x: x[0]*x[1]*x[2], even_ones))).T)
            print("\n ODD:")
            display(sp.Matrix(list(map(lambda x: x[0]*x[1]*x[2], odd_ones))).T)
            print("\n NEITHER")
            display(sp.Matrix(list(map(lambda x: x[0]*x[1]*x[2], faulty_ones))).T)

        bf_solution = {}
        for faulty in faulty_ones:
            print("Trying to fix", faulty)
            solution_pieces = list(set(permutations(faulty)))
            # Attempt to fix it by trying out all the sign choices for all the 
            # permutations of the faulty triple
            for fixing_signs in product(*([[-1,1]]*len(solution_pieces))):
                a_solution = dict(zip(solution_pieces, fixing_signs))
                Vg = V_coefficients.V_fixer(dict(V_coeff_original), a_solution, group, verbose=False)
                combo_types = Vg['combo_types']
                all_permutation_comparisons = Vg['all_permutation_comparisons']
                even_ones = [k for k,v in combo_types.items() if v == 'even']
                singular_ones = [k for k,v in combo_types.items() if v == 'singular']
                odd_ones = [k for k,v in combo_types.items() if v == 'odd']
                neither_ones = [k for k,v in combo_types.items() if v == 'neither']
                if faulty not in (neither_ones):
                    if verbose:
                        print("Found solution!")
                    bf_solution.update(a_solution)
                    break
            else:
                if verbose:
                    print("Found NO solution.")
                # one might surrender if no solution is found
                # for one triple
                # one might also keep on trying
                # and hope for a partial fix
                if not never_surrender:
                    break

        Vg = V_coefficients.V_fixer(dict(V_coeff_original), bf_solution, group, verbose=False)
        combo_types = Vg['combo_types']
        all_permutation_comparisons = Vg['all_permutation_comparisons']
        even_ones = [k for k,v in combo_types.items() if v == 'even']
        vee_symmetries[group_label]['even'] = even_ones
        singular_ones = [k for k,v in combo_types.items() if v == 'singular']
        vee_symmetries[group_label]['singular'] = singular_ones
        odd_ones = [k for k,v in combo_types.items() if v == 'odd']
        vee_symmetries[group_label]['oddd'] = odd_ones
        neither_ones = [k for k,v in combo_types.items() if v == 'neither']
        vee_symmetries[group_label]['neither'] = neither_ones
        final_bad_ones[group_label] = len(neither_ones)
        if verbose:
            print("\n EVEN:")
            display(sp.Matrix(list(map(lambda x: x[0]*x[1]*x[2], even_ones))).T)
            print("\n ODD:")
            display(sp.Matrix(list(map(lambda x: x[0]*x[1]*x[2], odd_ones))).T)
            print("\n NEITHER")
            display(sp.Matrix(list(map(lambda x: x[0]*x[1]*x[2], neither_ones))).T)
        if len(neither_ones) == 0:
            success[group_label] = True
        else:
            success[group_label] = False
        better_Vs[group_label] = dict(Vg)['V_coeff']
    print('+'*20)
    return better_Vs, vee_symmetries

if __name__ == '__main__':
    better_Vs, vee_symmetries = refine_CGs(verbose=False)
    if save_to_pickle:
        print("Saving V coeffients to pickle %s" % pickle_name)
        pickle.dump(better_Vs,open(pickle_name,'wb'))
        print("Saving their symmetries to pickle %s" % vee_syms_pickle)
        pickle.dump(vee_symmetries, open(vee_syms_pickle,'wb'))

