#!/usr/bin/env python

import os, re, pickle
import numpy as np
import sympy as sp
from textwrap import wrap
from time import time
from math import floor, log10, ceil
from qdef.constants import *
from functools import reduce
from sympy.combinatorics.permutations import Permutation

module_dir = os.path.dirname(__file__)

def unit_vec(size, pos):
    vec = [0]*size
    vec[pos] = 1
    return vec

def inverse_diagonal_eye(dim):
    return sp.Matrix(list(map(lambda x: x[-1::-1], sp.eye(dim).tolist())))

def list_splitter(seq, idx):
    '''
    Splits a sequence in three parts: (tail), element, and (head).
    Sequences are either list, tuples, or strings.

    Parameters
    ----------
    seq (seq): a sequence.
    idx (int): the index of the split position. Not checked.

    Returns
    -------
    (tail, element, head) (tuple) with:
        tail (seq): seq[:idx]
        elem (obj): seq[idx]
        head (seq): seq[idx+1:]
    '''
    return seq[:idx], seq[idx], seq[idx+1:]

def latex_float(afloat, num_decimal_digits = 2, latex_ready = True) -> str:
    '''
    Given  a float, return a latex representation with the required number
    of  decimal  digits after the decimal point, using standard scientific
    notation.
    '''
    template = "{0:.%de}" % num_decimal_digits
    float_str = template.format(afloat)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        exponent = int(exponent)
        if exponent == 0:
            template = '%.'+str(num_decimal_digits)+'f'
            if latex_ready:
                return '$%s$' % (template % afloat)
            else:
                return template % afloat
        else:
            if latex_ready:
                return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
            else:
                return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        if latex_ready:
            return '$%s$' % str(afloat)
        else:
            return str(afloat)

def rational_approx(x, N, min=0):
    '''
    Given  a number x this function returns a fraction
    that approximates it with a denominator that could
    be as large as N.
    '''
    if abs(x) <= min:
        return sp.S(0)
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

def inspector_gadget(small_seq, seq_of_seqs):
    '''
    This  function takes a tuple (or list) and a tuple (or list) of tuples
    (or  lists)  and  returns the index of the element of seq_of_seqs that
    has the same elements as small_seq.

    Parameters
    ----------
    small_seq (list or tuple)
    seq_of_seqs (list or tuple of lists or tuples)
    Returns
    -------
    (idx, signature) (tuple) where:
        idx (int): index of seq_of_seqs that mathces
        signature  (-1  or  1):  signature of the permutation that orders the
        elements of small_seq to match the element of seq_of_seqs.
    '''
    for idx, anElement in enumerate(seq_of_seqs):
        if set(anElement) == set(small_seq):
            perm = Permutation([small_seq.index(bq) for bq in anElement])
            return idx, perm.signature()
    else:
        return None

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
            simpler = rational_approx(v, n)
            # If the thing was approximated to zero
            # escalate the precision.
            while simpler == 0:
                n = 10*n
                simpler = rational_approx(v, n)
            sympy_dict[k] = simpler
    total = sum([k*v for k,v in sympy_dict.items()])
    return total

def square_rational_approx(x, N,min=0):
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
    squared_approx = sign*sp.sqrt(rational_approx(y,N,min))
    return squared_approx

def clarify_rep(string_rep):
    '''
    Replaces letter exponents to integers (as strings).
    '''
    RADIX_ATLANTE = ''.maketrans(dict(zip('abcdefghijklmnopqrstuvwxyz'.upper(),list(map(str,range(10,36))))))
    string_rep = string_rep.strip()
    string_rep = string_rep.translate(RADIX_ATLANTE)
    return string_rep

def prime_parser(string_rep):
    '''
    From string to quasirational.
    Assumes that expansion in primes goes up to the 12-th prime.
    Parameters
    ----------
    string_rep (str): string representation of a quasirational number
    '''
    NUM_PRIMES = 12
    PRIMES = list(map(lambda x:sp.S(sp.prime(x)), range(1,NUM_PRIMES+1)))
    string_rep = clarify_rep(string_rep)
    num_rep = list(map(int, string_rep.split(' ')))
    a0 = num_rep[0]
    if len(num_rep) == 1:
        num_rep = a0
    else:
        tail = sp.sqrt(reduce(sp.core.mul.Mul,[p**a for p,a in zip(PRIMES,num_rep[1:]) if a!=0]))
        num_rep = a0 * tail
    return num_rep


def double_group_matrix_inverse(double_group_chartable, group_label):
    '''
    Sympy's matrix inversion routine fails or stalls for some of
    the  groups.  This  is  a workaround by using numpy's matrix
    inversion  plus some simplifications to coerce every element
    of the matrix inverse to have a square that is rational.

    This  approximation  returns  an  exact matrix inverse as is
    guaranteed by the assert statement before the return line.
    '''
    # manual magical fix
    if group_label == 'O_{h}':
        N = 10000
    else:
        N = 1000
    chartable = double_group_chartable.T
    num_rows = chartable.rows
    num_cols = chartable.cols
    chararray = np.array(chartable).astype(np.complex128)
    chararrayinverse = np.linalg.inv(chararray)
    charinverse = sp.N(sp.Matrix(chararrayinverse),chop=True)
    simple_inverse = []
    num_rows = charinverse.rows
    num_cols = charinverse.cols
    for row_idx in range(num_rows):
        row = [sp.I*square_rational_approx(sp.im(charinverse[row_idx,col_idx]),N,1e-6) + square_rational_approx(sp.re(charinverse[row_idx,col_idx]),N,1e-6) for col_idx in range(num_cols)]
        simple_inverse.append(row)
    simple_inverse = sp.Matrix(simple_inverse)
    check = simple_inverse*chartable
    check = sp.re(check) + sp.I * sp.im(check)
    check = sp.N(check, chop=True)
    elements = (set(list(check)))
    # this guarantees that the matrix inverse is exact:
    unit_check = (len(elements) == min(2, num_rows) and (list(sorted(list(elements))) == [0,1]))
    assert unit_check, "ERROR in matrix inversion"
    return simple_inverse

def fmt_table(data, center_data=False, add_row_nums=False):
    '''Create a LaTeX table from a given list of lists'''
    buf='''
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{%s}
'''
    max_cols = max(len(r) for r in data)
    column_spec = '|' + '|'.join(['c']*max_cols) + '|'
    buf = buf % column_spec
    row_idx = 0
    for row_data in data:
        row = ''
        if add_row_nums and row_idx > 0:
            row += str(row_idx) + " & "
        if center_data and row_idx > 0:
            to_add = ceil( (max_cols - len(row_data))/2 )
            row += ' & '.join([''] * to_add)
        row += ' & '.join([sp.latex(thing) for thing in row_data])
        if row_idx == 0:
            row = '''\\hline ''' + row + '''\\\\\hline '''
        else:
            row += '''\\\\\hline '''
        row += "\n"
        buf +=row
        row_idx += 1
    buf += '''\\end{array}'''
    return buf

def latex_eqn_to_png(tex_code, timed=True, figname=None, outfolder=os.path.join(module_dir,'images')):
    '''
    Parse a given equation into a png and pdf formats.
    
    Requires a local installation of pdflatex and convert.
    
    Parameters
    ----------
    tex_code  (str): An equation, including $$, \[\] or LaTeX equation environment.
    time     (bool): If True then the filename includes a timestamp.
    figname   (str): If given, filenams for png and pdf have that root.
    outfolder (str): Directory name where the images will be stored.

    Returns
    -------
    None
    
    Example
    -------
    
    tex_code = r"""\begin{equation}
    x+y
    \end{equation}"""
    
    latex_eqn_to_png(tex_code, True, 'simple_eqn', outfolder = '/Users/juan/')
    
    --> creates /Users/juan/simple_eqn.pdf and /Users/juan/simple_eqn.npg
    
    Nov-24 2021-11-24 14:39:28
    '''
    now = int(time())
    temp_folder = '/Users/juan/Temp/'
    if outfolder == None:
        outfolder = temp_folder
    temp_latex = os.path.join(temp_folder,'texeqn.tex')
    header = r'''\documentclass[border=2pt]{standalone}
    \usepackage{amsmath}
    \usepackage{varwidth}
    \begin{document}
    \begin{varwidth}{\linewidth}'''
    footer = '''\end{varwidth}
    \end{document}'''
    texcode = "%s\n%s\n%s" % (header, tex_code, footer)
    open(temp_latex,'w').write(texcode)
    os.system('cd "%s"; /Library/TeX/texbin/pdflatex "%s"' % (temp_folder,temp_latex))
    os.system('cd "%s"; /opt/homebrew/bin/convert -density 300 texeqn.pdf -quality 90 texeqn.png' % (temp_folder))
    os.system(('cd "%s";' % (temp_folder)) + '/opt/homebrew/bin/convert texeqn.png -fuzz 1\% -trim +repage texeqn.png')
    os.system('cd "%s"; rm texeqn.log  texeqn.tex texeqn.aux' % (temp_folder))
    if timed and figname == None:
        out_pdf_fname = os.path.join(outfolder,"texeqn-%d.pdf" % now)
        out_png_fname = os.path.join(outfolder,"texeqn-%d.png" % now)
    elif timed and figname != None:
        out_pdf_fname = os.path.join(outfolder,"%s-%d.pdf" % (figname,now))
        out_png_fname = os.path.join(outfolder,"%s-%d.png" % (figname,now))
    elif not timed and figname == None:
        out_pdf_fname = os.path.join(outfolder,"texeqn.pdf")
        out_png_fname = os.path.join(outfolder,"texeqn.png")
    elif not timed and figname != None:
        out_pdf_fname = os.path.join(outfolder,"%s.pdf" % (figname))
        out_png_fname = os.path.join(outfolder,"%s.png" % (figname))
    print(out_pdf_fname)
    os.system('cd "%s"; mv texeqn.pdf "%s"' % (temp_folder, out_pdf_fname))
    os.system('cd "%s"; mv texeqn.png "%s"' % (temp_folder, out_png_fname))
    return None

def fixwidth(words,w):
    '''
    Receive  a  string and return a set of lines whose
    width  is less than or equal to the provided width
    w.  To  achieve  this,  additional whitespaces are
    added  as necessary. This function protects double
    newlines  but  puts  together  everything  that is
    separated by single newlines.
    '''
    pars = words.split('\n\n')
    chunks = []
    for chunk in pars:
        par = ' '.join(chunk.split('\n'))
        par = re.sub(' +',' ',par)
        lines = wrap(par,width=w)
        for lineidx, line in enumerate(lines[:-1]):
            if lineidx == len(lines):
                continue
            if len(line) == w:
                continue
            chars = list(line)
            if ' ' in chars:
                space_indices = [i for i, x in enumerate(chars) if x == ' ']
                idx = 0
                while len(''.join(chars)) < w:
                    space_index = space_indices[idx]
                    chars[space_index] = chars[space_index] + ' '
                    idx = (idx+1) % len(space_indices)
            elif ',' in chars:
                comma_indices = [i for i, x in enumerate(chars) if x == ',']
                idx = 0
                while len(''.join(chars)) < w:
                    comma_index = comma_indices[idx]
                    chars[comma_index] = chars[comma_index] + ' '
                    idx = (idx+1) % len(comma_indices)
            elif '.' in chars:
                comma_index = chars.index('.')
                chars[comma_index] = '.' + ' '*(w-len(chars))
            lines[lineidx] = ''.join(chars)
        chunks.append('\n'.join(lines))
    return '\n\n'.join(chunks)

def simple_formula_parser(job):
    '''
    Parse  a  chemical  formula  into  a dictionary of
    element  strings  and quantities. Formula must not
    contain parenthesis.
    '''
    sform = job[0]
    mult = job[1]
    formula = {}
    formula['original'] = sform
    sform = re.sub(r'([1-9]+)',r' \1 ',sform)
    sform = re.sub(r'([a-z])([A-Z])','\\1 1 \\2',sform)
    sform = re.sub(r'([A-Z])([A-Z])','\\1 1 \\2',sform)
    if sform[-1] not in '1 2 3 4 5 6 7 8 9'.split(' '):
        sform = sform+' 1'
    sform = sform.strip()
    parsed = sform.split(' ')
    parsed = dict(list(zip(parsed[::2],list(map(lambda x: int(x)*mult,parsed[1::2])))))
    formula['parsed'] = parsed
    return {k:v for k,v in parsed.items() if k !=''}

def complex_formula_parser(sform):
    '''
    Parse  a  chemical  formula into a list of dictionaries that
    specify elements in keys and number of atoms as values.

    The  different dictionaries relate to the different parts of
    the formula as parsed from the parenthesis structure.

    -------
    Example

    complex_formula_parser(Ca(CO2)2)   returns  a  list  of  two
    dictionaries  one  for  the single calcium atom, and another
    for the oxalate part of the formula like so:

       --> [{'C': 2, 'O': 4}, {'Ca': 1}]
    '''

    if sform[-1] == ')':
        sform = sform + '1'
    if ')' in sform:
        multipliers = list(map(int,re.findall('\)(.)',sform)))
        groups = list(zip(re.findall(r'\((.*?)\)',sform), multipliers))
        # delete every single digit after a right parenthesis
        sform = re.sub(r'\)[0-9]',r')',sform)
        # delete every thing inside parentheses
        sform = re.sub(r'\(.+\)','',sform)
        groups = groups + [(sform,1)]
    else:
        groups = [(sform,1)]
    return list(map(simple_formula_parser,groups))

def roundsigs(num, sig_figs):
    '''
    Round to a given number of significant figures.
    '''
    try:
        sign = np.sign(num)
        num = num*sign
        sciform = np.format_float_scientific(num).lower()
        power = int(sciform.split('e')[-1])
        mant = np.round(float(sciform.split('e')[0]), decimals=sig_figs-1)
        return sign*mant*(10**power)
    except:
        return np.nan

def labeled_matrix(a_matrix, basis_labels=[], elbow='', show=True):
    '''
    Parameters
    ----------
    a_matrix (sp.Matrix of list): square matrix of nested list.
    basis_labels (list[(str)])  : list of string representing labels for the basis kets.
    display (bool): if True then the resulting LaTeX expression is displayed.
    elbow (str) : text display in the top left corner, preferable a math expression.

    If basis_labels is empty then a sequence of integers are used for the labels.
    Returns
    -------
    a_matrix (sp.Matrix) : matrix with added row and column to represent the basis labels.
    '''
    from IPython.display import display, Math
    if isinstance(a_matrix, sp.Matrix):
        a_matrix = a_matrix.tolist()
    if len(basis_labels) == 0:
        basis_labels = list(range(len(a_matrix)))
    num_rows = len(a_matrix)
    assert len(a_matrix) == len(a_matrix[0]), "Matrix is not square."
    assert len(basis_labels) == len(a_matrix), "Basis and matrix dimensions mismatch."
    basis_kets = [sp.Symbol( '|{%s}\\rangle' % bl) for bl in basis_labels]
    basis_bras = [sp.Symbol('\\langle{%s}|' % bl) for bl in basis_labels]
    a_matrix = [[bk]+row for bk, row in zip(basis_bras,a_matrix)]
    extra_row = [sp.Symbol(elbow)]+basis_kets
    a_matrix = sp.latex(sp.Matrix([extra_row]+a_matrix))
    a_matrix = a_matrix.replace('c'*num_rows,('c'*num_rows).replace('c','c|',1)).replace('\\\\','\\\\[0.2cm] \\hline \\\\[-0.2cm] ', 1).replace('\\left[','',1).replace('\\right]','')
    if show:
        display(Math(a_matrix))
    return a_matrix
class UnitCon():
    '''
    Primary  conversion  factors  in  ConversionFactors.xlsx. To
    regenerate   the   fully   connected   network   run  script
    unitexpander.py.
    '''
    conversion_facts = pickle.load(open(os.path.join(
                            module_dir,
                            'data',
                            'conversion_facts.pkl'),'rb'))
    hc = 1239.8424749521
    @classmethod
    def con_factor(cls, source_unit, target_unit):
        '''
        This function returns how much of the target_unit
        is in the source_unit.

        For Angstrom use Å.

        --------
        Examples

        UnitCon.con_factor('Kg','g') -> 1e-3
        UnitCon.con_factor('Ha','Ry') -> 2.0
        '''
        if source_unit == target_unit:
            return 1.
        elif (source_unit, target_unit) in cls.conversion_facts.keys():
            return cls.conversion_facts[(source_unit, target_unit)]
        else:
            raise ValueError('This conversion I know not of.')

class Con():
    '''
    Fundamental and derived constants, all given in SI units.
    c, π, ℏ, h, ε0, μ0
    '''
    c = 299792458 # m/s
    π = 3.14159265358
    h = 6.62607015e-34 # Js
    ℏ = h / (2 * π) # Js
    ε0 = 8.8541878128e-12 # F/m
    μ0 = 1.25663706212e-6 # H/m
    e = 1.602176634e-19 # C
    me = 9.10938370e-31 # kg
    μB = e*ℏ/2/me # J/T
    gs = 2.002319304362

# =============================================================== #
# ======================= File Management =====+================= #

def split_dump(obj, rootname, folder='.', max_size_in_MB=99):
    '''
    Serialize an object into files of a given maximum size.
    Parameters
    ----------
    obj  (any object that can be pickled):
    rootname (str): base for filenames, not including extension or folder
    folder   (str): relative or absolute target directory
    max_size_in_MB (int): max size in MB for the pickled parts.
    Returns
    -------
    None
    '''
    max_size_in_MB = int(max_size_in_MB)
    max_size = max_size_in_MB*1024**2
    fnametemplate = '%s-%d.part.pkl'
    match_pattern = '%s-[\d]+.part.pkl' % rootname
    current_matches = list(filter(lambda s: re.match(match_pattern, s), os.listdir(folder)))
    current_matches_fnames = [os.path.join(folder,match) for match in current_matches]
    for match in current_matches_fnames:
        # a safety measure
        if '.part.pkl' not in match:
            raise Exception("Since this is deleting stuff, best check to see what happened here.")
        os.remove(match)
    else:
        if len(current_matches):
            print("Removed %d previous matching files." % (len(current_matches)))
    obj_bytes = pickle.dumps(obj)
    byte_len = len(obj_bytes)
    if byte_len > max_size:
        chunk_size = max_size
        counter, cursor = 0, 0
        while True:
            if cursor > byte_len:
                break
            fname = os.path.join(folder, fnametemplate % (rootname, counter))
            lcursor, rcursor = cursor, cursor + chunk_size
            with open(fname,'wb') as file:
                file.write(obj_bytes[lcursor:rcursor])
            counter += 1
            cursor = rcursor
        print("Object pickled to %d file(s)." % (counter))
    else:
        fname = os.path.join(folder, rootname+'.pkl')
        with open(fname,'wb') as file:
            pickle.dump(obj, file)
        print("Object pickled to a single file.")
    return None

def split_load(rootname, folder='.'):
    '''
    Reconstitute a serialized object that was split into several chunks
    using split_dump.
    Parameters
    ----------
    obj  (any object that can be pickled)
    rootname (str): root for filenames, not including extension or folder.
    folder   (str): relative or absolute target directory
    max_size_in_MB (int): max size in MB for the pickled parts.
    Returns
    -------
    object
    '''
    fnametemplate = rootname + '-%d.part.pkl'
    match_pattern = '%s-[\d]+.part.pkl' % rootname
    current_matches = list(filter(lambda s: re.match(match_pattern, s), os.listdir(folder)))
    if len(current_matches) > 0:
        print("Rebuilding file from %d parts" % len(current_matches))
        bites = bytes()
        for id in range(len(current_matches)):
            fname = os.path.join(folder,fnametemplate % id)
            with open(fname,'rb') as file:
                bites += file.read()
        return pickle.loads(bites)
    else:
        print("No parts found matching given rootname and folder.")
        singler = os.path.join(folder, rootname+'.pkl')
        if os.path.exists(singler):
            print("Found unsplit alt.")
            return pickle.load(open(singler, 'rb'))
        else:
            return None

# ======================= File Management =====+================= #
# =============================================================== #
