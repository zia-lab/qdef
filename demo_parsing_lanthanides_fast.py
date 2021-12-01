#!/usr/bin env python

import re
from sympy.parsing.latex import parse_latex
import sympy as sp
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl
import pickle
from random import random
from math import log10, floor
import time

table_fname = './data/lanthanide_tables/HFEnergyMatrixTables'
pickle_fname = './data/lanthanide_tables/HFEnergyMatrixTables.pkl'
save_to_pickle = True
pretty_vars = False # whether the vars are standard or nice

base_rep=[
('B02',sp.Symbol('B_{0,2}')),('B04',sp.Symbol('B_{0,4}')),
('B06',sp.Symbol('B_{0,6}')),('B0x',sp.Symbol('B_{0,x}')),
('B0y',sp.Symbol('B_{0,y}')),('B0z',sp.Symbol('B_{0,z}')),
('B12',sp.Symbol('B_{1,2}')),('B14',sp.Symbol('B_{1,4}')),
('B16',sp.Symbol('B_{1,6}')),('B22',sp.Symbol('B_{2,2}')),
('B24',sp.Symbol('B_{2,4}')),('B26',sp.Symbol('B_{2,6}')),
('B34',sp.Symbol('B_{3,4}')),('B36',sp.Symbol('B_{3,6}')),
('B44',sp.Symbol('B_{4,4}')),('B46',sp.Symbol('B_{4,6}')),
('B56',sp.Symbol('B_{5,6}')),('B66',sp.Symbol('B_{6,6}')),
('E0',sp.Symbol('E_{0}')),('E1',sp.Symbol('E_{1}')),
('E2',sp.Symbol('E_{2}')),('E3',sp.Symbol('E_{3}')),
('eOrbitalRad',sp.Symbol(r'\epsilon')),('gI',sp.Symbol('g_{I}')),
('gs',sp.Symbol('g_{s}')),('M0',sp.Symbol('M_{0}')),
('M2',sp.Symbol('M_{2}')),('M4',sp.Symbol('M_{4}')),
('P2',sp.Symbol('P_{2}')),('P4',sp.Symbol('P_{4}')),
('P6',sp.Symbol('P_{6}')),('S12',sp.Symbol('S_{1,2}')),
('S14',sp.Symbol('S_{1,4}')),('S16',sp.Symbol('S_{1,6}')),
('S22',sp.Symbol('S_{2,2}')),('S24',sp.Symbol('S_{2,4}')),
('S26',sp.Symbol('S_{2,6}')),('S34',sp.Symbol('S_{3,4}')),
('S36',sp.Symbol('S_{3,6}')),('S44',sp.Symbol('S_{4,4}')),
('S46',sp.Symbol('S_{4,6}')),('S56',sp.Symbol('S_{5,6}')),
('S66',sp.Symbol('S_{6,6}')),('\[Alpha]',sp.Symbol(r'\alpha')),
('\[Beta]',sp.Symbol(r'\beta')),('\[Beta]BohrMag',sp.Symbol(r'\mu_{B,e}')),
('\[Beta]n',sp.Symbol(r'\mu_{B,n}')),('\[Gamma]',sp.Symbol(r'\gamma')),
('\[Zeta]',sp.Symbol(r'\zeta'))]

master_rep = {}
for idx, it in enumerate(base_rep):
    key = sp.Symbol('x_{%d}' % (idx+1))
    master_rep[key] = it[1]
inverse_rep = {v:k for k,v in master_rep.items()}

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
ToSympy[expr0_] := (
  expr = Expand[Chop[expr0]];
  expr = expr /. reps;
  str = ToString[FullForm[expr, NumberMarks -> False]];
  str = StringReplace[
    str,
    {"Plus" -> "sp.core.add.Add",
     "Times" -> "sp.core.mul.Mul",
     "Power" -> "sp.core.power.Pow",
     "[" -> "(", "]" -> ")",
     "List" -> "slist",
     "\"" -> ""}];
  str = StringReplace[str,
    {"Subscript(x" -> "sp.SubscriptSymbol('x'",
     "Rational" -> "sp.Rational"}];
  Return[str]
  )
''')

# Abbreviations to simplify parsing
if pretty_vars:
    def SubscriptSymbol(a,b):
        return master_rep[sp.Symbol("%s_{%d}" % (a,int(b)))]
else:
    def SubscriptSymbol(a,b):
        return sp.Symbol("%s_{%d}" % (a,int(b)))
sp.Rational = lambda x,y: sp.S(x)/sp.S(y)
sp.SubscriptSymbol = SubscriptSymbol
Pi = sp.pi
Complex = lambda x,y: (sp.S(x) + sp.I * sp.S(y))
def slist(*args):
    return list([*args])

def parse_mathematica(mathematica_expression):
    seval = session.evaluate('ToSympy[%s]' % mathematica_expression)
    return str(seval)

def lanthanum_cleanup(fname):
    '''
    Data  file  might have large redundancies, this opens it and removes
    all    redundant   definitions;   it   assumes   that  there are  no
    inconsistencies between them, if there are then it fails.
    
    More  importantly  it puts together all the lines that relate to one
    definition in just one string with no newlines. 

    This is assuming that the file only contains definitions for:
       EnergyMatrixTable, AllowedM, and EnergyStatesTable

    Parameters
    ----------
    fname   (str): file name of file to be parsed

    Returns
    -------
    clean_output  (list): a list of strings each with a single definition.
    '''
    lanthanum = [l.strip().split(' =')[0] for l in open(fname,'r').readlines() if l[0] != ' ']
    lanthanum = list(filter(lambda x: x != '', lanthanum))
    # this dictionary will have as keys the lhs of definitions
    # and as values will be lists of strings that all attempt
    # to define this symbol
    rhs = {}
    full_lanthanum = [l.strip() for l in open(fname,'r').readlines()]
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
    # now to see if there are any redundancies
    clean_output = []
    for k,v in rhs.items():
        unique = list(set(v))
        assert len(unique) == 1, "There should be only one, this is a loopy file."
        unique = unique[0]
        out = '%s = %s' % (k, unique)
        clean_output.append(out)
    return clean_output
def parse_table(fname, verbose=False):
    clear_lanthanum = lanthanum_cleanup(fname)
    EnergyMatrixTables = {}
    for cl in clear_lanthanum:
        if 'EnergyMatrixTable' in cl:
            pre = cl.split(' =')[0]
            if verbose:
                print("Parsing:",pre)
            parse = parse_mathematica(cl)
            args = tuple(eval(re.findall(r'\[.*\]',pre)[0]))
            EnergyMatrixTables[args] = sp.Matrix(eval(parse))
    EnergyStatesTable = {}
    for cl in clear_lanthanum:
        if 'EnergyStatesTable' in cl:
            lhs = cl.split(' =')[0]
            lhs = tuple(eval(re.findall(r'\[.*\]',lhs)[0]))
            rhs = cl.split('= ')[1]
            parse = parse_mathematica(cl)
            parse = parse.replace('3P','"3P"').replace('1S','"1S"').replace('3F','"3F"')\
                    .replace('1D','"1D"').replace('1G','"1G"').replace('3H','"3H"')\
                    .replace('1I','"1I"')
            EnergyStatesTable[lhs] = eval(parse)
    AllowedM = {}
    for cl in clear_lanthanum:
        if 'AllowedM' in cl:
            parse = parse_mathematica(cl)
            AllowedM[parse[0]] = eval(parse)
    return {'EnergyMatrixTables': EnergyMatrixTables,
            'EnergyStatesTable': EnergyStatesTable,
            'AllowedM': AllowedM}

if __name__ == '__main__':
    if pretty_vars:
        print("Parsing into matrices with pretty variables B_{0,x}, E_{0}, ...")
    else:
        print("Parsing into standard subindexed vars x_1, x_2 ..")
    start_time = time.time()
    parsed_table = parse_table(table_fname)
    if save_to_pickle:
        print("Saving to pickle...")
        pickle.dump(parsed_table,open(pickle_fname,'wb'))
        print("Pickle saved!")
    print("Time elapsed = %d s" % int(time.time() - start_time))
    session.stop()
