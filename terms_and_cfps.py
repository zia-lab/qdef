#!/usr/bin/env python3

import re
import sympy as sp
import pandas as pd
from collections import OrderedDict as odict
from collections import namedtuple as ntuple
from functools import reduce
from notation import *
import pickle
import os
from fractions import Fraction
from collections import Counter

# ┌─────────────────────────────────────────┐
# │                                         │
# │     This script parses datasets of      │
# │ spectroscopic terms, seniority numbers, │
# │and coefficients of fractional parentage.│
# │                                         │
# │               1650580062                │
# │                                         │
# └─────────────────────────────────────────┘

RADIX_ATLANTE = ''.maketrans(dict(zip('abcdefghijklmnopqrstuvwxyz'.upper(),list(map(str,range(10,36))))))
NUM_PRIMES = 12
PRIMES = list(map(lambda x:sp.S(sp.prime(x)), range(1,NUM_PRIMES+1)))
body_nums = [1,2,3,4]

export = True
cfps_folder = './data'

# named tuples
State = ntuple('State',['seniority','sup_irrep'])
Term = ntuple('Term',['S','L','W'])
CFP = ntuple('CFP',['daughter', 'parent_n_minus_b','parent_b'])
Config = ntuple('Config',['l','n'])

cfp_template = ntuple('cfp',['bod','l',
                            'n_daughter', 'S_daughter', 'L_daughter', 'W_daughter',
                            'n_parent_1', 'S_parent_1', 'L_parent_1', 'W_parent_1',
                            'n_parent_2', 'S_parent_2', 'L_parent_2', 'W_parent_2'])
s_Label = ntuple('d_Term',['l','n','label','multiS','L','idx','seniority','S'])
d_Label = ntuple('d_Term',['l','n','label','multiS','L','idx','seniority','R5','S'])
f_Label = ntuple('d_Term',['l','n','label','multiS','L','idx','seniority','R7','G2','variant','S'])

def clarify_rep(string_rep):
    '''
    Replaces letter exponents to integers (as strings).
    '''
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
    string_rep = clarify_rep(string_rep)
    num_rep = list(map(int, string_rep.split(' ')))
    a0 = num_rep[0]
    if len(num_rep) == 1:
        num_rep = a0
    else:
        tail = sp.sqrt(reduce(sp.core.mul.Mul,[p**a for p,a in zip(PRIMES,num_rep[1:]) if a!=0]))
        num_rep = a0 * tail
    return num_rep

if __name__ == '__main__':
    print("> Parsing term labels from Nielson and Koster ...")

    term_lines = open('./data/pdf_terms.txt','r').read().split('\n')
    term_labels = odict()
    for line in term_lines:
        if line == '' or line[0] == '#':
            continue
        if line[0] == '-':
            l = l_notation_switch(line[1])
            n = int(line[2])
            term_labels[(l,n)] = []
            continue
        if l == 1:
            term_label = line.split(' ')[0]
            multiS = sp.S(int(line[0]))
            S = Fraction((multiS-1)/2)
            L = l_notation_switch(line[1])
            seniority = int(line.split(' ')[1])
            term_labels[(l,n)].append(s_Label(l,n,term_label,multiS,L,1,seniority,S))
        elif l == 2:
            term_label = line.split(' ')[0]
            multiS = sp.S(int(line[0]))
            S = Fraction((multiS-1)/2)
            L = l_notation_switch(line[1])
            seniority = int(line.split(' ')[1])
            R5 = tuple(map(int,line.split(' ')[2]))
            if len(term_label) == 3:
                idx = int(term_label[2])
            else:
                idx = 1
            term_labels[(l,n)].append(d_Label(l,n,term_label,multiS,L,idx,seniority,R5,S))
        elif l == 3:
            term_label = line.split(' ')[0]
            multiS = sp.S(int(line[0]))
            S = Fraction((multiS-1)/2)
            L = l_notation_switch(line[1])
            seniority = int(line.split(' ')[1])
            R7 = tuple(map(int,line.split(' ')[2]))
            G2 = tuple(map(int,line.split(' ')[3]))
            if len(line.split(' ')) == 5:
                variant = line[-1]
            else:
                variant = 'A'
            if len(term_label) >= 3:
                idx = int(term_label[2:])
            else:
                idx = 1
            term_labels[(l,n)].append(f_Label(l,n,term_label,multiS,L,idx,seniority,R7,G2,variant,S))

    sorted_term_labels = odict()
    for k in term_labels:
        sorted_labels = list(sorted(term_labels[k], key=lambda x:x[5]))
        sorted_labels = list(sorted(sorted_labels, key=lambda x:x[4]))
        sorted_labels = list(sorted(sorted_labels, key=lambda x:x[3]))
        sorted_term_labels[k] = sorted_labels

    dframes = [pd.DataFrame(list(map(lambda x: x._asdict(),sorted_term_labels[k]))) for k in sorted_term_labels]
    print(">> Creating dataframe of spectroscopic terms ...")
    termFrame = pd.concat(dframes).reset_index(drop=True)

    rows = [x for x in zip(termFrame['l'], termFrame['n'], termFrame['multiS'], termFrame['L'],termFrame['R5'],termFrame['R7'], termFrame['G2'])]
    counts = Counter([x for x in zip(termFrame['l'], termFrame['n'], termFrame['multiS'], termFrame['L'],termFrame['R5'],termFrame['R7'], termFrame['G2'])])
    count_col = [counts[row] for row in rows]
    termFrame['variants'] = count_col
    termFrame['l^n'] = ['%s^%d' % (l_notation_switch(x),y) for x,y in zip(termFrame['l'],termFrame['n'])]

    termFrame['multiL'] = [2*x+1 for x in termFrame['L']]
    termFrame['multiLS'] = [x*y for x,y in zip(termFrame['multiS'],termFrame['multiL'])]
    cols = 'l n l^n label S L idx multiS multiL multiLS seniority R5 R7 G2 variants variant'.split(' ')
    termFrame = termFrame[cols]

    new_labels = []
    for label, idx in zip(termFrame['label'], termFrame['idx']):
        if not label[-1].isdigit():
            new_labels.append(label+'1')
        else:
            new_labels.append(label)
    termFrame['extra_label'] = new_labels

    print(">> Parsing dictionary of term seniority ...")
    seniority_dict = [((l,n,sp.S(S),L,idx),senior) for l,n,S,L,idx,senior in zip(termFrame['l'], termFrame['n'], termFrame['S'], termFrame['L'], termFrame['idx'], termFrame['seniority'])]
    seniority_dict = dict(seniority_dict)

    if export:
        print(">> Exporting ...")
        termFrame.to_csv('./data/all_term_labels.csv')
        termFrame.to_pickle('./data/all_term_labels.pkl')
        termFrame.to_excel('./data/all_term_labels.xlsx')
        pickle.dump(seniority_dict, open('./data/seniority_dict.pkl','wb'))

    print("> Parsing minimal set of coefficients of fractional parentage from Velkov's tables ...")

    big_frame = []
    for body_num in body_nums:
        body_cfps_dict = {}
        for l in [1,2,3]:
            print('>>', (body_num, l_notation_switch(l).upper()))
            fname = os.path.join(cfps_folder,'B%d%s_ALL.txt' % (body_num, l_notation_switch(l).upper()))
            if not os.path.exists(fname):
                continue
            lines = open(fname,'r').read().split('\n')
            cfps = odict()
            cfps_prime = odict()
            for line in lines:
                oline = line
                if line == '' or line[0] == '/':
                    continue
                if line[0] == '[':
                    sector = line.split(']')[0][-2:]
                    sector = Config(l_notation_switch(sector[0]),int(sector[1]))
                    cfps[sector] = odict()
                    cfps_prime[sector] = odict()
                    continue
                if '[DAUGHTER TERM]' in line:
                    dau = line.split('[')[0]
                    assert len(dau) <= 4
                    if not dau[-1].isdigit():
                        W_dau = 1
                    else:
                        W_dau = int(dau[2:])
                    L_dau, SMulti_dau =  l_notation_switch(dau[1]), int(dau[0])
                    S_dau = sp.S(SMulti_dau-1)/2
                    term_dau = Term(S_dau, L_dau, W_dau)
                    n_daughter = sector.n
                    n_parent_1 = n_daughter - body_num
                    n_parent_2 = body_num
                else:
                    line = line.replace('-',' -')
                    rline = re.sub(' +', ' ', line.strip())
                    rline = rline.split(' ')
                    parent_n_minus_bod = rline[0]
                    assert len(parent_n_minus_bod) <= 4
                    if not parent_n_minus_bod[-1].isdigit():
                        W_par_n_minus_bod = 1
                    else:
                        W_par_n_minus_bod = int(parent_n_minus_bod[2:])
                    L_par_n_minus_bod, SMulti_par_n_minus_bod = l_notation_switch(parent_n_minus_bod[1]), int(parent_n_minus_bod[0])
                    S_par_n_minus_bod = sp.S(SMulti_par_n_minus_bod-1)/2
                    term_parent_1 = Term(S_par_n_minus_bod, L_par_n_minus_bod, W_par_n_minus_bod)
                    parent_bod = rline[1]
                    assert len(parent_bod) <= 4
                    if not parent_bod[-1].isdigit():
                        W_par_bod = 1
                    else:
                        W_par_bod = int(parent_bod[2:])
                    L_par_bod, SMulti_par_bod = l_notation_switch(parent_bod[1]), int(parent_bod[0])
                    S_par_bod = sp.S(SMulti_par_bod-1)/2
                    
                    term_parent_2 = Term(S_par_bod, L_par_bod, W_par_bod)

                    cfp = ' '.join(rline[2:])
                    cfp_num = prime_parser(cfp)
                    cfps[sector][CFP(term_dau, term_parent_1, term_parent_2)] = cfp_num

                    row = (body_num, sector.l,
                        n_daughter, *term_dau,
                        n_parent_1, *term_parent_1,
                        n_parent_2, *term_parent_2, cfp_num)
                    big_frame.append(row)
                    key = (body_num, sector.l, 
                        n_daughter, *term_dau, 
                        n_parent_1, *term_parent_1, 
                        n_parent_2, *term_parent_2)
                    body_cfps_dict[key] = cfp_num
        if export:
            dict_fname = os.path.join('./data/','CFP_%d-body-dict.pkl' % body_num)
            print(">> Saving %s..." % dict_fname)
            pickle.dump(body_cfps_dict, open(dict_fname,'wb'))
    dframe_fname = os.path.join('./data/','CFP_all-dframe.pkl')
    col_labels = 'bod l n_daughter S_daughter L_daughter W_daughter n_parent_1 S_parent_1 L_parent_1 W_parent_1 n_parent_2 S_parent_2 L_parent_2 W_parent_2 CFP'.split(' ')
    CFP_frame = pd.DataFrame(big_frame, columns=col_labels)
    if export:
        CFP_frame.to_pickle(dframe_fname)
    print("> Finished.")