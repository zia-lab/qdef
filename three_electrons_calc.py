#!/usr/bin/env python3

######################################################################
#                            _____                                   #
#                           |__  /      ___                          #
#                            /_ <______/ _ \                         #
#                          ___/ /_____/  __/                         #
#                         /____/      \___/                          #
#                                                                    #
#                                                                    #
######################################################################

# Dec-13 2021-12-13 15:51:48

import sys
import sympy as sp
import numpy as np
from qdef import *
from misc import *
from itertools import product, permutations
from sympy import Eijk as εijk
import pickle, os

new_labels = pickle.load(open('./Data/components_rosetta.pkl','rb'))
save_to_pickle = True
pickle_fname = os.path.join('./','data','three_electrons.pkl')
if os.path.exists(pickle_fname) and save_to_pickle:
    reply = input('%s already exists, overwrite y/n? ' % pickle_fname)
    if reply == 'n':
        sys.exit()

def det_simplify(ket_parts):
    equivalent_parts = {}
    standard_order = {} # this holds the standard order to which all the other list members will be referred to
    for ket_part_key, ket_part_coeff in ket_parts.items():
        set_ket = frozenset(ket_part_key)
        if set_ket not in equivalent_parts:
            equivalent_parts[set_ket] = []
            standard_order[set_ket] = ket_part_key
        equivalent_parts[set_ket].append((ket_part_key, ket_part_coeff))
    # once I've grouped them together into pices of equivalent parts
    # i then need to rearrange and properly sign the rearrangements
    det_simple = []
    for equivalent_key in equivalent_parts:
        base_order = standard_order[equivalent_key]
        total_coeff = 0
        equiv_parts = equivalent_parts[equivalent_key]
        for equiv_part in equiv_parts:
            ordering = [base_order.index(part) for part in equiv_part[0]]
            sign = εijk(*ordering)
            total_coeff += sign*equiv_part[1]
        final = (base_order, total_coeff)
        if total_coeff != 0:
            det_simple.append(final)
    return det_simple

def qet_divide(qet0, qet1):
    '''
    Given   two   qets,   assumed   to   be   superpositions  of
    determinantal states. Determine if they are collinear and if
    they are, provide their ratio.

    Parameters
    ----------
    qet0    (qdef.Qet) : a qet with determinantal keys.
    qet1    (qdef.Qet) : a qet with determinantal keys.

    Returns
    -------
    ratio (num): 0 if qets are not collinear, otherwise equal to
    qet0/qet1.


    '''
    if len(qet0.dict) != len(qet1.dict):
        return 0
    set0 = frozenset(map(frozenset,qet0.dict.keys()))
    set1 = frozenset(map(frozenset,qet1.dict.keys()))
    num_parts = len(qet0.dict)
    # a necessary condition for them to be possibly collinear
    # is that they should have have the same sets of quantum
    # numbers.
    if set0 != set1:
        return 0
    else:
        ratios = []
        # iterate over the quantum nums of the first qet
        for qet_key_0, qet_val_0 in qet0.dict.items():
            set0 = set(qet_key_0)
            # and determine the ratio that it has
            # to all of the parts of the other qet
            # allowing for reaarangmenets valid
            # under determinantal state rules
            for qet_key_1, qet_val_1 in qet1.dict.items():
                set1 = set(qet_key_1)
                if set0 == set1:
                    ordering = [qet_key_0.index(qk) for qk in qet_key_1]
                    sign = εijk(*ordering)
                    ratios.append(sign * qet_val_0/qet_val_1)
                    continue
    if ratios == []:
        return 0
    else:
        # if all of the ratios are equal
        # then the ratio of the two qets
        # is well defined
        if len(set(ratios)) == 1 and len(ratios) == num_parts:
            return ratios[0]
        else:
            return 0

def three_electron_waves(group_label, Γ1, Γ2, Γ3, disambiguate = False):
    '''
    Given  a  label for a crystallographic point group and three
    labels   for   corresponding   irreducible  representations.
    Produce  all the three electron wavefunctions  assuming that
    all the electrons are equivalent.

    Parameters
    ----------
    group_label     (str):
    Γ1        (sp.Symbol): label for an irrep of G
    Γ2        (sp.Symbol): label for an irrep of G
    Γ3        (sp.Symbol): label for an irrep of G
    disambiguate   (bool): if verbose output is returned

    Returns
    -------
    if disambiguate == False:
        final_qets      (dict): 
            keys -> (Γ_total, γ_total, 2*S_total+1, M_total, S_12, Γ_12)
            values -> qdef.Qet with determinantal states
    if disambiguate == True then also final_qets_groups is returned
        final_qets_groups (dict):
            keys -> (Γ_total, γ_total, 2*S_total+1, M_total, S_12, Γ_12)
            values -> dict() with keys 'paths' and 'qet'
                with 'paths' a list with all alternate intermediate two-
                electron states that lead to the same final qet.
    
    Where  S_12  is  the  spin  of the intermediate two-electron
    wavefunction and Γ_12 its corresponding irrep.
    '''
    # get the group
    group = CPGs.get_group_by_label(group_label)
    # get the CGs
    group_CGs = group.CG_coefficients
    # enforce the better notation for components
    flat_labels = dict(sum([list(l.items()) for l in list(new_labels[group_label].values())],[]))
    group_CGs = {(flat_labels[k[0]], flat_labels[k[1]], flat_labels[k[2]]):v for k,v in group_CGs.items()}

    irreps = group.irrep_labels
    s_half = sp.S(1)/2
    ms = mrange(s_half)

    component_labels = {k:list(v.values()) for k,v in new_labels[group_label].items()}
    comps_1, comps_2, comps_3 = [component_labels[ir] for ir in [Γ1, Γ2, Γ3]]
    s1, s2, s3 = s_half, s_half, s_half

    Γ12s = group.product_table.odict[(Γ1, Γ2)] # the intermediate irreps belong to the reduction of Γ1 X Γ2
    S12s = lrange(s1,s2) # this is just [0,1] as for the total angular momentum of the intermediate states
    total_kets = {}

    counter = 0
    for γ1, γ2, γ3, m1, m2, m3, Γ12, S12 in product(comps_1, comps_2, comps_3, ms, ms, ms, Γ12s, S12s):
        Γs = group.product_table.odict[(Γ12, Γ3)]
        Ss = lrange(S12, s3)
        comps_12 = component_labels[Γ12]
        M12s = mrange(S12)
        for Γ in Γs:
            comps = component_labels[Γ]
            for S in Ss:
                Ms = mrange(S)
                for γ, M12, M, γ12 in product(comps, M12s, Ms, comps_12):
                    total_ket_key = (Γ, γ, 2*S+1, M, S12, Γ12)
                    # summing s1 and s2 to yield S12
                    sCG1 = clebschG.eva(s1, s2, S12, m1, m2, M12)
                    # summing S12 and s3 to give the final S, M
                    sCG2 = clebschG.eva(S12, s3, S, M12, m3, M)
                    # coupling a γ1, γ2 to get a γ12
                    gCG1 = group_CGs.setdefault((γ1, γ2, γ12), 0)
                    # coupling a γ12, γ3 to get a final γ
                    gCG2 = group_CGs.setdefault((γ12, γ3, γ), 0)
                    coeff = sCG1 * sCG2 * gCG1 * gCG2
                    if coeff == 0:
                        continue
                    # collect in the dictionary all the parts that correspond to the sums
                    if total_ket_key not in total_kets:
                        total_kets[total_ket_key] = {}
                    if m1 < 0:
                        γ1f = bar_symbol(γ1)
                    else:
                        γ1f = γ1
                    if m2 < 0:
                        γ2f = bar_symbol(γ2)
                    else:
                        γ2f = γ2
                    if m3 < 0:
                        γ3f = bar_symbol(γ3)
                    else:
                        γ3f= γ3
                    total_ket_part_key = (γ1f, γ2f, γ3f)
                    if total_ket_part_key not in total_kets[total_ket_key]:
                        total_kets[total_ket_key][total_ket_part_key] = 0
                    total_kets[total_ket_key][total_ket_part_key] += coeff
                    counter += 1

    # The resulting qets may have redundant parts in that 
    # a rearrangment of the quantum symbols may be equal
    # to another part.
    # This makes it necessary to do some rearrangment
    # juggling, and permutations signing.

    simplified_kets = {}
    for total_ket_key, ket_parts in total_kets.items():
        det_simplified = det_simplify(ket_parts)
        if len(det_simplified) != 0:
            simplified_kets[total_ket_key] = Qet(dict(det_simplified))
            the_normalizer = 1/simplified_kets[total_ket_key].norm()
            simplified_kets[total_ket_key] = the_normalizer*simplified_kets[total_ket_key]

    final_qets = []
    corresponding_waves = []
    for total_ket_key_0, simple_ket_0 in simplified_kets.items():
        ratios = []
        for simple_ket_1 in final_qets:
            divvy = qet_divide(simple_ket_0, simple_ket_1)
            ratios.append(divvy==0)
        ratios = sum(ratios)
        if ratios == len(final_qets):
            final_qets.append(simple_ket_0)
            corresponding_waves.append(total_ket_key_0)
    final_qets = list(zip(corresponding_waves, final_qets))

    # in addition to removing the redundant ones
    # I would also like to keep al the total_ket_keys that correspond to each
    # group of redundant qets
    if disambiguate:
        final_qets_groups = {}
        for main_key, sqet in final_qets:
            final_qets_groups[main_key] = {'paths':[],'qet':sqet}
            for total_ket_key_0, simple_ket_0 in simplified_kets.items():
                if qet_divide(sqet, simple_ket_0) != 0:
                    final_qets_groups[main_key]['paths'].append(total_ket_key_0)
        return final_qets, final_qets_groups
    else:
        return final_qets

if __name__ == '__main__':
    group_label = 'O'
    group = CPGs.get_group_by_label(group_label)
    irreps = group.irrep_labels
    three_irreps_combos = list(combinations_with_replacement(irreps,3))
    all_three_electron_funs = {}
    print(len(three_irreps_combos))
    for idx, irrep_combo in enumerate(three_irreps_combos):
        print(idx,end='|')
        sys.stdout.flush()
        all_three_electron_funs[irrep_combo] = three_electron_waves(group_label,*irrep_combo)
    if save_to_pickle:
        print("\nSaving to pickle ...")
        pickle.dump(all_three_electron_funs, open(pickle_fname,'wb'))
