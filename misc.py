#!/usr/bin/env python

import pickle
import os, re
import numpy as np
from textwrap import wrap
from time import time
from math import floor,log10
import sympy as sp

module_dir = os.path.dirname(__file__)

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
    Basis constants, all given in standard SI units.
    c, π, ℏ, h, ε0, μ0
    '''
    c = 299792458
    π = 3.14159265358
    ℏ = 6.62607015e-34
    h = ℏ * 2 * π
    ε0 = 8.8541878128e-12
    μ0 = 1.25663706212e-6
