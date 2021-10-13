#!/usr/bin/env python

import pickle
import os, re
import numpy as np
from textwrap import wrap

module_dir = os.path.dirname(__file__)

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
