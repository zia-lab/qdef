#!/usr/bin/env python

######################################################################
#                                                                    #
#                  ____  ____ ______________  __________             #
#                 / __ \/ __ `/ ___/ ___/ _ \/ ___/ ___/             #
#                / /_/ / /_/ / /  (__  )  __/ /  (__  )              #
#               / .___/\__,_/_/  /____/\___/_/  /____/               #
#              /_/                                                   #
#                                                                    #
######################################################################


import re
from sympy.parsing.latex import parse_latex
import sympy as sp
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl

session=WolframLanguageSession()

session.evaluate(r'''vars = {B02, B04, B06, B0x, B0y, B0z, B12, B14, B16, B22, B24, B26, 
   B34, B36, B44, B46, B56, B66, E0, E1, E2, E3, eOrbitalRad, gI, gs, 
   M0, M2, M4, P2, P4, P6, S12, S14, S16, S22, S24, S26, S34, S36, 
   S44, S46, S56, 
   S66, \[Alpha], \[Beta], \[Beta]BohrMag, \[Beta]n, \[Gamma], \
\[Zeta]};
svars = Table[
   ToExpression[SubscriptBox["x", ToString[i]]], {i, 1, Length[vars]}];
reps = (#[[1]] -> #[[2]]) & /@ Transpose[{vars, svars}];
ParseSymbol[thing_] :=
 (str = ToString[Chop[thing] /. reps, TeXForm];
  Return[str]
  )''')

base_rep=[
('B02',sp.Symbol('B_{0,2}')),
('B04',sp.Symbol('B_{0,4}')),
('B06',sp.Symbol('B_{0,6}')),
('B0x',sp.Symbol('B_{0,x}')),
('B0y',sp.Symbol('B_{0,y}')),
('B0z',sp.Symbol('B_{0,z}')),
('B12',sp.Symbol('B_{1,2}')),
('B14',sp.Symbol('B_{1,4}')),
('B16',sp.Symbol('B_{1,6}')),
('B22',sp.Symbol('B_{2,2}')),
('B24',sp.Symbol('B_{2,4}')),
('B26',sp.Symbol('B_{2,6}')),
('B34',sp.Symbol('B_{3,4}')),
('B36',sp.Symbol('B_{3,6}')),
('B44',sp.Symbol('B_{4,4}')),
('B46',sp.Symbol('B_{4,6}')),
('B56',sp.Symbol('B_{5,6}')),
('B66',sp.Symbol('B_{6,6}')),
('E0',sp.Symbol('E_{0}')),
('E1',sp.Symbol('E_{1}')),
('E2',sp.Symbol('E_{2}')),
('E3',sp.Symbol('E_{3}')),
('eOrbitalRad',sp.Symbol(r'\epsilon')),
('gI',sp.Symbol('g_{I}')),
('gs',sp.Symbol('g_{g}')),
('M0',sp.Symbol('M_{0}')),
('M2',sp.Symbol('M_{2}')),
('M4',sp.Symbol('M_{4}')),
('P2',sp.Symbol('P_{2}')),
('P4',sp.Symbol('P_{4}')),
('P6',sp.Symbol('P_{6}')),
('S12',sp.Symbol('S_{1,2}')),
('S14',sp.Symbol('S_{1,4}')),
('S16',sp.Symbol('S_{1,6}')),
('S22',sp.Symbol('S_{2,2}')),
('S24',sp.Symbol('S_{2,4}')),
('S26',sp.Symbol('S_{2,6}')),
('S34',sp.Symbol('S_{3,4}')),
('S36',sp.Symbol('S_{3,6}')),
('S44',sp.Symbol('S_{4,4}')),
('S46',sp.Symbol('S_{4,6}')),
('S56',sp.Symbol('S_{5,6}')),
('S66',sp.Symbol('S_{6,6}')),
('\[Alpha]',sp.Symbol(r'\alpha')),
('\[Beta]',sp.Symbol(r'\beta')),
('\[Beta]BohrMag',sp.Symbol(r'\mu_{B}')),
('\[Beta]n',sp.Symbol(r'\beta_{n}')),
('\[Gamma]',sp.Symbol(r'\gamma')),
('\[Zeta]',sp.Symbol(r'\zeta'))];
master_rep = {}
for idx, it in enumerate(base_rep):
    key = sp.Symbol('x_{%d}' % (idx+1))
    master_rep[key] = it[1]

def lanthanum_cleanup(fname):
    lanthanum = [l.strip().split(' =')[0] for l in open(fname,'r').readlines() if l[0] != ' ']
    lanthanum = list(filter(lambda x: x != '', lanthanum))
    rhs = {}
    full_lanthanum = [l.strip() for l in open('/Volumes/GoogleDrive/My Drive/Zia Lab/Codebase/qdef/data/lanthanide_tables/HFEnergyMatrixTables copy 2','r').readlines()]
    for line_idx, line in enumerate(full_lanthanum):
        if line_idx < len(full_lanthanum)-1:
            next_line = full_lanthanum[line_idx+1]
        if 'Attributes[Null]' in line:
            continue
        if ('EnergyMatrixTable' in line) or ('AllowedM' in line) or ('EnergyStatesTable' in line):
            key = line.split('=')[0].strip()
            chunks = []
            try:
                first_chunk = line.split('=')[1]
            except:
                first_chunk = ''
            chunks.append(first_chunk)
            if key not in rhs:
                rhs[key] = []
            continue
        if line == '' or ('EnergyMatrixTable' in next_line) or ('AllowedM' in next_line) or ('EnergyStatesTable' in next_line):
            chunks.append(line.strip())
            whole_chunk = ''.join(chunks).strip()
            if whole_chunk != '':
                rhs[key].append(whole_chunk)
            chunks = []
        else:
            chunks.append(line.strip())
    clean_output = []
    for k,v in rhs.items():
        unique = list(set(v))
        assert len(unique) == 1
        unique = unique[0]
        out = '%s = %s' % (k, unique)
        clean_output.append(out)
    return clean_output

def parse_it(astr):
    astr = cleanup(astr)
    return sp.expand((parse_latex(astr)))

def cleanup(astr):
    reps = [('. ',''),
            ]
    for rep in reps:
        astr = astr.replace(*rep)
    return astr

def parse_energy_matrix_table(astr):
    lhs = astr.split('= ')[0].strip()
    rhs = astr.split('= ')[-1]
    # define it in the Mathematica session
    session.evaluate(astr)
    # get num rows
    num_rows, num_cols = tuple(session.evaluate("Dimensions[%s]" % lhs))
    parsed_matrix = []
    for num_row in range(1,num_rows+1):
        row = []
        for num_col in range(1, num_cols+1):
            parse = cleanup(session.evaluate("ParseSymbol[%s[[%d,%d]]]" % (lhs, num_row, num_col)))
            parsed = sp.expand(parse_latex(parse)).subs(sp.Symbol('i'),sp.I)
            row.append(parsed)
        parsed_matrix.append(row)
    return lhs, sp.Matrix(parsed_matrix)

def parse_allowed_m(astr):
    lhs = astr.split('= ')[0].strip()
    rhs = astr.split('= ')[-1]
    # define it in the Mathematica session
    session.evaluate(astr)
    # get num rows
    num_lists = tuple(session.evaluate("Dimensions[%s]" % lhs))[0]
    M_value = int(lhs.split('[')[-1].split(']')[0])
    rows = []
    for list_index in range(1,num_lists+1):
        the_list = session.evaluate("%s[[%d]]" % (lhs, list_index))
        try:
            the_row = [sp.S(x[0])/sp.S(x[1]) for x in the_list]
        except:
            the_row = [sp.S(x) for x in the_list]
        rows.append(the_row)
    return M_value, rows  

def parse_energy_states_table(astr):
    lhs = astr.split('= ')[0].strip()
    rhs = astr.split('= ')[-1]
    # define it in the Mathematica session
    session.evaluate(astr)
    # get num rows
    num_rows, num_cols = tuple(session.evaluate("Dimensions[%s]" % lhs))
    parsed_matrix = []
    rows = []
    for num_row in range(1,num_rows+1):
        head = session.evaluate("%s[[%d]][[-1]]" % (lhs, num_row))
        try:
            head = sp.S(head[0])/sp.S(head[1])
        except:
            head = sp.S(head)
        thorax = session.evaluate("%s[[%d]][[1]][[-1]]" % (lhs, num_row))
        try:
            thorax = sp.S(thorax[0])/sp.S(thorax[1])
        except:
            thorax = sp.S(thorax)
        knees = session.evaluate("%s[[%d]][[1]][[1]]" % (lhs, num_row))
        knees = (str(knees[0]), sp.S(knees[1]))
        row = (((knees),thorax),head)
        rows.append(row)
    args = '(%s)' % lhs.split('[')[-1].split(']')[0]
    return sp.S(args), rows

def parse_lanthanides_tablefile(fname):
    clear_lanthanum = lanthanum_cleanup(fname)
    EnergyMatrixTables = {}
    for cl in clear_lanthanum:
        if 'EnergyMatrixTable' in cl:
            print('.',end='|')
            parse = parse_energy_matrix_table(cl)
            args = sp.S(parse[0].split('[')[-1].split(']')[0])
            EnergyMatrixTables[args] = parse[1].subs(master_rep)
    EnergyStatesTable = {}
    for cl in clear_lanthanum:
        if 'EnergyStatesTable' in cl:
            print('.',end='|')
            parse = parse_energy_states_table(cl)
            args = parse[0]
            EnergyStatesTable[args] = parse[1]
    AllowedM = {}
    for cl in clear_lanthanum:
        if 'AllowedM' in cl:
            print('.',end='|')
            parse = parse_allowed_m(cl)
            AllowedM[parse[0]] = parse[1]
    return {'EnergyMatrixTables': EnergyMatrixTables,
            'EnergyStatesTable': EnergyStatesTable,
            'AllowedM': AllowedM}

