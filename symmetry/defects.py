#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coded by David in Feb of 2021

import pymatgen as mg
import numpy as np
from pymatgen.symmetry.analyzer import PointGroupAnalyzer


def calculate_vacancy_complexes(structure, dopant, tol=0.001):
    '''This function takes the structure of a unit cell and computes the
    possible vacancy complexes and their point group symmetry.

    Parameters:
    structure (pymatgen.core.structure.Structure): unit cell of the crystal
    dopant    (str): element symbol for the substitutional atom
    tol     (float): tolerance distance for the inclusion of nearest neighbors

    Returns:
    list: a list of lists where the first element is a string equal to the
          corresponding point group, and the second element equals a
          pymatgen.core.structure.Molecule molecule

    '''

    # compute the coordinates and species of all the atoms in the unit cell
    coords = np.array([[atom.x, atom.y, atom.z] for atom in structure.sites])
    species = [atom.name for atom in structure.species]
    # initialize a list where defect structures will be saved
    defects = []
    num_sites = len(species)
    structure_vol = structure.volume
    structure_r = structure_vol**(1/3)
    # - iterate through each of the unit cell atoms
    # - replace each by the given dopant
    # - find atoms within a sphere with volume equal to the unit cell vol
    # - find the nearest neighbors
    # - replace each of them by a vacancy, one at a time,
    # - determine what point group symmetry results
    for i in range(num_sites):
        dopant_coord = coords[i]
        this_defect = structure.get_sites_in_sphere(dopant_coord, structure_r)
        this_species = [atom[0].specie.name for atom in this_defect]
        this_coords = np.array([atom[0].coords for atom in this_defect])
        # index might be different, find the new one
        new_index = np.arange(len(this_coords))[np.all(this_coords == dopant_coord, axis=1)][0]
        # change the atom to the given dopant
        this_species[new_index] = dopant
        # find distances to other atoms
        distances = np.array([np.linalg.norm(dopant_coord - coord) for coord in this_coords])
        # find the nearest neighbors
        minDistance = distances[np.argsort(distances)[1]]
        neighbors = np.arange(len(this_coords))[np.abs(distances-minDistance) < tol]
        # replace each of them by a vacancy, one at a time
        for neighbor in neighbors:
            replacing = this_species[neighbor]
            this_species[neighbor] = 'V'
            a_defect = mg.core.structure.Molecule(this_species, this_coords)
            finder = PointGroupAnalyzer(a_defect)
            pg = str(finder.get_pointgroup())
            defects.append([pg, a_defect])
            # put the original atom back in the list of species
            this_species[neighbor] = replacing
    return defects
