#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coded by David in Feb of 2021

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
import pickle, os, sys
import pandas as pd
from decimal import Decimal
from pathlib import Path
from qdef.crystal_secrets import secrets

data_dir = os.path.join(Path(__file__).parent,'data')

polar_point_groups = '1 2 m mm2 3m 3 3m 4 4mm 6 6mm'.split(' ')
all_at_symbols = ['Ac','Ag','Al','Am','Ar','As','At','Au','B','Ba','Be','Bh',
                  'Bi','Bk','Br','C','Ca','Cd','Ce','Cf','Cl','Cm','Cn','Co',
                  'Cr','Cs','Cu','Db','Ds','Dy','Er','Es','Eu','F','Fe','Fl',
                  'Fm','Fr','Ga','Gd','Ge','H','He','Hf','Hg','Ho','Hs','I',
                  'In','Ir','K','Kr','La','Li','Lr','Lu','Lv','Mc','Md','Mg',
                  'Mn','Mo','Mt','N','Na','Nb','Nd','Ne','Nh','Ni','No','Np',
                  'O','Og','Os','P','Pa','Pb','Pd','Pm','Po','Pr','Pt','Pu',
                  'Ra','Rb','Re','Rf','Rg','Rh','Rn','Ru','S','Sb','Sc','Se',
                  'Sg','Si','Sm','Sn','Sr','Ta','Tb','Tc','Te','Th','Ti','Tl',
                  'Tm','Ts','U','V','W','Xe','Y','Yb','Zn','Zr']
ferrenti_gt = pd.read_pickle(os.path.join(data_dir,'ferrenti_gt.pkl'))
all_spinless = pd.read_json(os.path.join(data_dir,'all_spinless.json'))



mpdr = MPDataRetrieval(api_key=secrets['materials_project_api_key'])

def some_in_list(l1,l2):
    return sum([L1 in l2 for L1 in l1]) > 0

def spinless(abundance):
    '''This function returns a list of atoms with
    at least the provided spinless natural abundance'''
    spin_less_and_plenty = all_spinless[all_spinless['nat_spinless_abundance']>=abundance]
    return spin_less_and_plenty.index.tolist()
def spinfull(abundance):
    spineless = spinless(abundance)
    return [s for s in all_at_symbols if s not in spineless]

def crystal_sieve(sieve_parameters):
    if sieve_parameters['element_selection'][0] == 'natural':
        return crystal_sieve_natural(sieve_parameters)
    elif sieve_parameters['element_selection'][0] == 'manual':
        return crystal_sieve_manual(sieve_parameters)
    else:
        print("Invalid choice for element selection.")
        return None
def crystal_sieve_natural(sieve_parameters):
    spinless_abundance = sieve_parameters['element_selection'][1]
    min_gap, max_gap = sieve_parameters['bandgap_interval']
    num_ingredients = list(range(sieve_parameters['num_ingredients'][0],
                                 sieve_parameters['num_ingredients'][1]+1))
    ####################################################################
    print("Filtering for elements with a natural spinless isotopic abundance equal or greater than %.1f%%." % (100*spinless_abundance))
    sieve_parameters['allowed_elements'] = spinless(spinless_abundance)
    sieve_parameters['not_allowed_elements'] = spinfull(spinless_abundance)
    print("There are %d elements matching this criterion." % len(sieve_parameters['allowed_elements']))
    ####################################################################
    print("Searching the Materials Project...")
    criteria = {'band_gap':{'$gte':min_gap,'$lte':max_gap},
                'nelements': {'$in': num_ingredients},
                'elements':{'$in':sieve_parameters['allowed_elements'],
                            '$nin':sieve_parameters['not_allowed_elements']}}
    props = ['band_gap','cif','elements','pretty_formula',
             'density','nelements', 'structure','spacegroup',
            'e_above_hull','icsd_ids','formation_energy_per_atom']
    sieve_parameters['search_results'] = mpdr.get_dataframe(criteria=criteria,
                     properties=props)
    sieve_parameters['search_results']['polar'] = [s['point_group'] in polar_point_groups for s in sieve_parameters['search_results']['spacegroup']]
    sieve_parameters['search_results']['contains_unviable_elements'] = sieve_parameters['search_results']['elements'].apply(lambda x: some_in_list(x, sieve_parameters['unviable_elements']))
    return sieve_parameters
def crystal_sieve_manual(sieve_parameters):
    return "Hello manual."
