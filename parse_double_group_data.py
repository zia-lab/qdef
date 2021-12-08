#!/usr/bin/env python3

import sympy as sp
from sympy import I
from sympy.parsing.latex import parse_latex
import re
import pickle

# Parsing data for the double-valued representations into qdef.

# Mathematica notebook which generates the raw data is `qdef_gtpack.nb`

# Data for each group is parsed into a dictionary with the following keys:

# - `index`: int [1-32]
# - `notation`: str Mulliken or Bethe
# - `group label`: str eg. "C_{3v}"
# - `irrep labels`: list of sympy.Symbol
# - `class labels`: list of sympy.Symbol
# - `classes`: dict  {k(class - sympy.Symbol) : v(group-ops list[sympy.Symbol])}
# - `character table`: sympy.Matrix, each row corresponds to an irrep
# - `multiplication table`: sympy.Matrix, left factor in product determined by row
# - `euler angles`: dict { k(group_op - sympy.Symbol) : v([α, β, γ, det]) }
# - `group operations`: list of sympy.Symbol

# In addition, a set of metadata is also created here. It is saved as a dictionary with the following keys:

# - `Mulliken notation` : dictionary that specifies the letters used in Mulliken notation
# - `Euler angles` : explanation of convention used for Euler angles
# - `Rotation axes` : dict {k(axis label): [x,y,z]}

pickle_fname = './data/gtpackdata_double_groups.pkl'

mulliken_notation = {'A': 'singly degenerate rep; symmetrical for rotations about the principal axis',
                    'B': 'singly degenerate rep; anti-symmetrical for rotations about the principal axis',
                    'E': 'either doubly degenerate rep, or one of a pair of singly degenerate conjugate reps',
                    'T': 'triply degenerate rep',
                    'F': 'fourfold degenerate rep, or one of a pair of doubly degenerate conjugate reps',
                    'H': 'fivevold degenerate rep'}
mulliken_or_bethe_notation = 'In all groups Mulliken notation is used, except groups 11 and 23 where Bethe notation is used instead.'
rotation_axes = {
                'a': [1,1,0],
                'b': [1,-1,0],
                'c': [1,0,1],
                'd': [-1,0,1],
                'e': [0,1,1],
                'f': [0,1,-1],
                'γ': [1,-1,-1],
                'β': [-1,1,-1],
                'α': [-1,-1,1],
                'δ': [1,1,1],
                'x': [1,0,0],
                'y': [0,1,0],
                'z': [0,0,1],
                'A': [-sp.sqrt(3)/3,-1,0],
                'B': [-sp.sqrt(3)/3,1,0],
                'C': [sp.sqrt(3),-1,0],
                'D': [sp.sqrt(3),1,0]
                }
rotation_axes = {k: (sp.Matrix(v)/sp.Matrix(v).norm()).T.tolist()[0] for k, v in rotation_axes.items()}
euler_angles_note = '''First three elements are the angles, last element is the determinant of the rotation.
                    Active rotations are assumed.
                    first angle -> α: final rotation about the fixed z-axis
                    sencond angle -> β: intermediate rotation about the fixed y-axis
                    third angle -> γ: initial rotation about the fixed z-axis,
                    corresponding rotation matrix is:
                    [[-sin(α)*sin(γ) + cos(α)*cos(β)*cos(γ),-sin(α)*cos(γ) - sin(γ)*cos(α)*cos(β),sin(β)*cos(α)],
                    [sin(α)*cos(β)*cos(γ) + sin(γ)*cos(α),-sin(α)*sin(γ)*cos(β) + cos(α)*cos(γ),sin(α)*sin(β)],
                    [-sin(β)*cos(γ),sin(β)*sin(γ),cos(β)]]'''
metadata = {'Mulliken notation': mulliken_notation,
            'Euler angles': euler_angles_note,
            'Rotation axes': rotation_axes}

def parse_double_group_data(group_index):
    '''
    Parse the output from GTPack v 1.32, as produced in qdef.nb.
    '''
    def symbolize(symb):
        return sp.Symbol(symb)
    def simplify_pies(expr):
        return expr.subs(sp.Symbol("pi"),sp.pi)
    notation = 'Mulliken'
    # if group_index in [11,23]:
    #     notation = 'Bethe'
    fname = './Group Data/DoubleGroup_%d.txt' % group_index
    group_string = open(fname,'r').read()
    
    group_lines = group_string.split('\n')
    
    # line 0 has the index and the name of the group
    group_label = group_lines[0].split(',')[1]
    if len(group_label) > 1:
        group_label = 'D-%s_{%s}' % (group_label[0],group_label[1:])
    group_index = int(group_lines[0].split(',')[0])
    
    # line 1 has the labels for the irreducible representations
    irrep_labels = group_lines[1].split(',')
    irrep_labels = list(map(symbolize,irrep_labels))
    
    # line 2 has the dimensions of the irreducible representations
    irrep_dims = list(map(int,group_lines[2].split(',')))
    
    # line 3 has the sizes of the conjugacy classes
    class_dims = list(map(int,group_lines[3].split(',')))
    
    # line 4 has the elements of classes
    class_elements_flat = list(map(symbolize, group_lines[4].split(',')))
    num_elements = len(class_elements_flat)
    
    # line 5 has the labels for the classes
    class_labels = list(map(symbolize, group_lines[5].split(',')))
    num_classes = len(class_labels)

    # parse the classes as a dictionary
    post = 0
    classes = {}
    for class_label, class_dim in zip(class_labels,class_dims):
        classes[class_label] = class_elements_flat[post:post+class_dim]
        post += class_dim
    
    # line 6 has the labels for the group elements in an ordering matching the irreducible rep matrices ordering
    group_op_labels = list(map(symbolize, group_lines[6].split(',')))
    
    # line 7 has the euler angles for all the group operations
    eulerangles_raw = list(map(parse_latex, group_lines[7].split(',')))
    eulerangles = {}
    for idx, group_op_label in enumerate(group_op_labels):
        eulerangles[group_op_label] = list(map(simplify_pies,eulerangles_raw[4*idx:4*(idx+1)]))
    
    # line 8 has the character table
    char_table = list(map(lambda x: parse_latex(x).subs(sp.Symbol('i'),I), group_lines[8].split(',')))
    char_table = sp.Matrix(char_table).reshape(num_classes,num_classes)
    
    # line 9 has the multiplication table
    mult_table = group_lines[9].split(',')
    mult_table = sp.Matrix(list(map(symbolize,mult_table))).reshape(num_elements,num_elements)
    
    return {'index': group_index,
           'notation': notation,
           'group label': group_label,
           'irrep labels': irrep_labels,
           'class labels': class_labels,
        #    'irrep matrices': irrep_matrices,
           'classes': classes,
        #    'generators': generators,
           'class labels': class_labels,
           'character table': char_table,
           'multiplication table': mult_table,
           'euler angles': eulerangles,
           'group operations': group_op_labels}

if __name__ == '__main__':
    double_group_data = {i: parse_double_group_data(i) for i in range(1,33)}
    print("Saving to pickle...")
    pickle.dump({'metadata': metadata,
                'group_data': double_group_data},
            open(pickle_fname,'wb'))
