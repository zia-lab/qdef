#!/usr/bin/env python

import re
from sympy.parsing.latex import parse_latex
import sympy as sp
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl
import pickle
import time
from random import betavariate, random
from math import log10, floor
import sys

save_to_pickle = True
pickle_fname = './data/lanthanide_tables/HFEnergyMatrixTables.pkl'

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
 (str = ToString[Chop[thing] /. reps, CForm];
  Return[str]
  )''')

# NB: The order here needs to match the order
# of vars in the Mathematica assignment above

base_rep=[
('B02',sp.Symbol('B_{0,2}')), # x_1
('B04',sp.Symbol('B_{0,4}')), # x_2
('B06',sp.Symbol('B_{0,6}')), # ...
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
('gs',sp.Symbol('g_{s}')),
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
('\[Beta]BohrMag',sp.Symbol(r'\mu_{B,e}')),
('\[Beta]n',sp.Symbol(r'\mu_{B,n}')),
('\[Gamma]',sp.Symbol(r'\gamma')),
('\[Zeta]',sp.Symbol(r'\zeta'))]

master_rep = {}
for idx, it in enumerate(base_rep):
    key = sp.Symbol('x_{%d}' % (idx+1))
    master_rep[key] = it[1]
inverse_rep = {v:k for k,v in master_rep.items()}

def lanthanum_cleanup(fname):
    '''
    Data  file  might have large redundancies, this opens it and removes
    all    redundant   definitions;   it   assumes   that   there's   no
    inconsistencies between them, if there is then it fails.
    
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

def cleanup(parse):
    '''
    Some basic cleanup to avoid some errors
    when using parse_latex.
    '''
    parse = re.sub(r'Subscript\(x,(\d{1,2})\)', r'x_{\1}', parse)
    parse = re.sub(r'Complex\((.*?),(.*?)\)',r'(\1+i*(\2))',parse)
    parse = re.sub(r'Sqrt\((.*?)\)', r'sqrt{\1}', parse).replace('sqrt','\sqrt').replace('.*','*').replace('.)',')')
    parse = re.sub(r'(\d\.[\d]+)e([-]{0,1}\d)',r'(\1*10^{\2})', parse)
    parse = parse + ' '
    reps = [('. ','')]
    for rep in reps:
        parse = parse.replace(*rep)
    return parse

def parse_energy_matrix_table(astr):
    '''
    Parse a string that contains the definition for a symbolic matrix.

    Parameters
    ----------
    astr    (str): A definition of the sort EnergyMatrixTable[_,_,_,_,_] = {{...},{...},...}

    Returns
    -------
    lhs, parsed_matrix, rhs
    lhs                 (str): The LHS of the matrix definition
    parsed_matrix (sp.Matrix): The parsed symbolic matrix.
    rhs                 (str): The RHS of the matrix definition
    '''
    lhs = astr.split('= ')[0].strip()
    rhs = astr.split('= ')[-1]
    # define it in the Mathematica session
    session.evaluate(astr)
    # get num rows and columns
    num_rows, num_cols = tuple(session.evaluate("Dimensions[%s]" % lhs))
    parsed_matrix = []
    # Iterate through each element and parse it
    for num_row in range(1,num_rows+1):
        row = []
        for num_col in range(1, num_cols+1):
            parse = session.evaluate("ParseSymbol[%s[[%d,%d]]]" % (lhs, num_row, num_col))
            # Clean up the string before parsing with Sympy
            parse = cleanup(parse)
            parsed = sp.expand(parse_latex(parse)).subs(sp.Symbol('i'),sp.I)
            # The imaginary unit needs to be dealt with separately
            parsed = parsed.subs(sp.Symbol('i'),sp.I)
            row.append(parsed)
        parsed_matrix.append(row)
    parsed_matrix = sp.Matrix(parsed_matrix)
    return lhs, parsed_matrix, rhs

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

def parse_table(fname):
    '''
    Put everything together to parse
    EnergyMatrixTable, EnergyStatesTable, and AllowedM.

    Parameters
    ----------
    fname   (str): Filename of the file with the Mathematica defs.

    Returns
    -------
    {'EnergyMatrixTables': EnergyMatrixTables,
     'EnergyStatesTable': EnergyStatesTable,
     'AllowedM': AllowedM,
     'EnergyMatrixStrings': EnergyMatrixStrings}
    with
    EnergyMatrixTables    (dict): Keys are tuples () values are symbolic sp.Matrix
    EnergyStatesTable     (dict): Keys are tuples () values are lists
    AlloweM               (dict): Key is an integer corresponding to __, values are
    EnergyMatrixStrings   (dict): Keys are tuples () values are the original strings.

    '''
    clear_lanthanum = lanthanum_cleanup(fname)
    EnergyMatrixTables = {}
    EnergyMatrixStrings = {}
    counter = 0
    for cl in clear_lanthanum:
        if 'EnergyMatrixTable' in cl:
            # print('.',end='|')
            # sys.stdout.flush()
            pre = cl.split(' =')[0]
            print('Parsing: %s' % pre)
            parse = parse_energy_matrix_table(cl)
            args = sp.S(parse[0].split('[')[-1].split(']')[0])
            EnergyMatrixTables[args] = parse[1].subs(master_rep)
            EnergyMatrixStrings[args] = parse[2]
            counter += 1
    EnergyStatesTable = {}
    for cl in clear_lanthanum:
        if 'EnergyStatesTable' in cl:
            parse = parse_energy_states_table(cl)
            args = parse[0]
            EnergyStatesTable[args] = parse[1]
    AllowedM = {}
    for cl in clear_lanthanum:
        if 'AllowedM' in cl:
            parse = parse_allowed_m(cl)
            AllowedM[parse[0]] = parse[1]
    return {'EnergyMatrixTables': EnergyMatrixTables,
            'EnergyStatesTable': EnergyStatesTable,
            'AllowedM': AllowedM,
            'EnergyMatrixStrings': EnergyMatrixStrings}

def rational_simplify(sympy_expr, N=10000):
    '''
    Given a sympy expression this function takes it and
    finds rational  approximations (perhaps including a
    square root).

    Example
    -------

    >> rational_simplify(2.31099*sp.Symbol('x') - 1.14)
    >>> 9 * sqrt(546) * x / 91 - sqrt(130)/10
    '''
    sympy_dict = sympy_expr.as_coefficients_dict()
    for k,v in sympy_dict.items():
        if isinstance(v, sp.core.numbers.Float):
            n = N
            simpler = square_rational_approx(v, n)
            # If the thing was approximated to zero
            # escalate the precision.
            while simpler == 0:
                n = 10*n
                simpler = square_rational_approx(v, n)
            sympy_dict[k] = simpler
    total = sum([k*v for k,v in sympy_dict.items()])
    return total

def rational_approx(x, N):
    '''
    Given  a number x this function returns a fraction
    that approximates it with a denominator that could
    be as large as N.
    '''
    if (int(x) == x):
        return sp.S(int(x))
    sign = 1
    if x < 0:
        sign = -1
        x = -x
    if x > 1:
        ix, dx = int(x), x - int(x)
    else:
        ix = 0
        dx = x
    exponent = -floor(log10(float(dx)))
    tens_multiplier = int(exponent-1)
    dx = dx*(10**tens_multiplier)
    divider = 1/(sp.S(10)**(sp.S(tens_multiplier)))
    sign = sign
    a, b = 0, 1
    c, d = 1, 1
    while (b <= N and d <= N):
        mediant = float(a+c)/(b+d)
        if dx == mediant:
            if b + d <= N:
                return sign*(sp.S(ix)+divider*sp.S(a+c)/sp.S(b+d))
            elif d > b:
                return sign*(sp.S(ix)+divider*sp.S(c)/sp.S(d))
            else:
                return sign*(sp.S(ix)+divider*sp.S(a)/sp.S(b))
        elif dx > mediant:
            a, b = a+c, b+d
        else:
            c, d = a+c,b+d
    if (b > N):
        return sign*(divider*sp.S(c)/sp.S(d) + sp.S(ix))
    else:
        return sign*(divider*sp.S(a)/sp.S(b) + sp.S(ix))

def square_rational_approx(x, N):
    '''
    Given a number x this algorithm finds the best  rational
    approximation to its square, and then returns the signed
    square root of that.
    '''
    if x < 0:
        sign = -1
        x = -x
    else:
        sign = 1
    y = x*x
    return sign*sp.sqrt(rational_approx(y,N))

if __name__ == '__main__':
    start_time = time.time()
    parsed_table = parse_table('/Volumes/GoogleDrive/My Drive/Zia Lab/Codebase/qdef/data/lanthanide_tables/HFEnergyMatrixTables')
    print("Time elapsed = %d" % int(time.time() - start_time))
    print("Simplifying numeric coefficients ...")
    for k,v in parsed_table['EnergyMatrixTables'].items():
        num_rows = v.rows
        num_cols = v.cols
        for num_row in range(num_rows):
            for num_col in range(num_cols):
                v[num_row,num_col] = sp.expand(rational_simplify(v[num_row,num_col]))
    print("Time elapsed = %d" % int(time.time() - start_time))
    print("Validating parsing by stochastic evaluation ...")
    diffs = {}
    for k,v in parsed_table['EnergyMatrixTables'].items():
        free_symbs = v.free_symbols
        free_symbs_values = {v: 10*(random()-0.5) for v in free_symbs}
        mathematica_values = {inverse_rep[k]:v for k,v in free_symbs_values.items()}
        num_try = sp.N(v.subs(free_symbs_values))
        energyMatrixString = parsed_table['EnergyMatrixStrings'][k]
        mathematica_subs = ', '.join(['Subscript[x,%s] -> %s' % (str(str(k).split('{')[-1].split('}')[0]), str(v)) for k,v in mathematica_values.items()])
        mathematica_subs = '{%s}' % mathematica_subs
        mathematica_try = sp.Matrix(session.evaluate('Re[Chop[(%s /. reps)] /. %s]' % (energyMatrixString, mathematica_subs))) +\
                    sp.I*sp.Matrix(session.evaluate('Im[Chop[(%s /. reps)] /. %s]' % (energyMatrixString, mathematica_subs))) 
        diff_mat = (mathematica_try - num_try)
        mathematica_norm = mathematica_try.norm()
        if mathematica_norm == 0:
            diffs[k] = diff_mat.norm()
        else:
            diffs[k] = diff_mat.norm()/mathematica_norm
        print("%s: Δ = %s" % (str(k), diffs[k]))
    max_diff = max(diffs.values())
    print("Max difference = {:e}".format(max_diff))
    assert(max_diff < 1e-6)
    if save_to_pickle:
        print("Saving to pickle")
        pickle.dump(parsed_table,open(pickle_fname,'wb'))
    print("Time elapsed = %d" % int(time.time() - start_time))
    session.stop()