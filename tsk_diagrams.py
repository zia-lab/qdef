#!/usr/bin/env python3

from qdef import *
import matplotlib.pyplot as plt
from misc import *
import pickle
import sympy as sp
from adjustText import adjust_text
from collections import Counter
from notation import *

welcome = '''----------------------------------------------------------------------
This  script  generates  Tanabe-Sugano diagrams, it can run on a saved
binary  file  with  the  necessary  data  or  it can recompute it from
scratch.  It  can  also  generate  latex  output  that  may be used to
generate  a view of the created diagrams and matrix representations of
the Hamiltonian.
----------------------------------------------------------------------'''

recompute_data = False # recomputes but does not overwrite
all_terms_ref_file = './data/all_terms_trimmed.pkl'
create_figures = True
fig_template = './images/dummy-TSK-GT-%d.pdf'
parse_latex = False
output_latex_file = '/Users/juan/Library/Mobile Documents/com~apple~CloudDocs/iCloudFiles/Theoretical Division/TSK-diag.tex'
target_electrons = [2,3,4,5,6,7,8]

def make_multi_term_symbol(psi_key):
    electrons = psi_key.electrons
    if len(set(electrons)) == 1:
        multi_term = sp.Symbol(str(electrons[0]).lower())**len(electrons)
    else:
        e_term = '{}^%d%s' % (psi_key.terms[0][0]*2+1, psi_key.terms[0][1])
        t2_term = '{}^%d%s' % (psi_key.terms[1][0]*2+1, psi_key.terms[1][1])
        num_es = electrons.count(sp.Symbol('E'))
        num_ts = electrons.count(sp.Symbol('T_2'))
        if num_es == 1:
            e_term = 'e'
        else:
            e_term = 'e^%d(%s)' % (num_es, e_term)
        if num_ts == 1:
            t2_term = 't_2'
        else:
            t2_term = 't_2^%d(%s)' % (num_ts, t2_term)
        multi_term = sp.Symbol('{%s}{%s}' % (t2_term, e_term))
    return multi_term

def LS_allowed_terms(l:int,n:int) -> dict:
    '''
    Calculate the allowed terms in LS coupling for homogeneous configurations.
    Parameters
    ----------
    l (int): orbital angular momentum
    n (int): how many electrons
    Returns
    -------
    terms (dict) with keys equal to (2S+1) multiplicities and values
    equal to list of allowed angular momenta.
    '''
    def flatten(nlist):
        flist = []
        for elem in nlist:
            for it in elem:
                flist.append(it)
        return flist
    ls = [l]*n
    spins = [-1/2, 1/2]
    terminators = {0:'S',1:'P',2:'D',3:'F',4:'G',5:'H',6:'I',7:'K',8:'L',
                   9:'M',10:'N',11:'O',12:'Q',13:'R',14:'T',15:'U',16:'V',
                  17:'W',18:'X',19:'Y',20:'Z'}
    single_states = []
    mLs = list(range(-l,l+1))
    for mL in mLs:
        for mS in [-1/2,1/2]:
            single_states.append((mL,mS))
    configs = list(map(set,list(combinations(single_states,n))))
    MLs = range(-sum(ls),sum(ls)+1)
    spins = np.arange(-1/2*len(ls),1/2*len(ls)+1)
    microstates = {}
    for ML in MLs:
        subconfigs = [config for config in configs if sum([l[0] for l in list(config)]) == ML]
        for mtot in spins:
            thestates = [list(config)[:len(ls)*2] for config in subconfigs if sum([l[1] for l in list(config)])==mtot]
            if len(thestates) > 0:
                microstates[(ML,mtot)] = list(map(flatten,thestates))
            else:
                microstates[(ML,mtot)] = []
    # find the non-empty ones
    # from those pick the coordinates that are closest to the lower left corner
    # if it is possible to to diagonally to the upper right, then this is a boxy box
    # if not, then it is a rowy row
    # it might also be a columny col
    collections = []
    while True:
        non_empty = [[k,abs(MLs[0]-k[0])+abs(spins[0]-k[1])] for k in microstates.keys() if len(microstates[k])>0]
        if len(non_empty) == 0:
            break
        corner = non_empty[np.argsort([r[-1] for r in non_empty])[0]][0]
        if corner == (0,0):
            case = 'box'
            start = (0,0)
            end = (0,0)
        else:
            right = (corner[0]+1, corner[1])
            up = (corner[0], corner[1]+1)
            diag = (corner[0]+1, corner[1]+1)
            if up in microstates.keys():
                up_bool = len(microstates[up]) > 0
            else:
                up_bool = False
            if right in microstates.keys():
                right_bool = len(microstates[right]) > 0
            else:
                right_bool = False
            if diag in microstates.keys():
                diag_bool = len(microstates[diag]) > 0
            else:
                diag_bool = False
            if diag_bool and up_bool and right_bool:
                case = 'box'
                start = corner
                end = (-corner[0], -corner[1])
            elif up_bool and not right_bool:
                case = 'col'
                start = corner
                end = (corner[0],-corner[1])
            else:
                case = 'row'
                start = corner
                end = (-corner[0], corner[1])
        if case == 'row':
            collect = []
            for k in np.arange(start[0], end[0]+1):
                collect.append(microstates[(k,0)].pop())
        elif case == 'col':
            collect = []
            for k in np.arange(start[1], end[1]+1):
                collect.append(microstates[(start[0],k)].pop())
        elif case == 'box':
            collect = []
            for k in np.arange(start[0], end[0]+1):
                for l in np.arange(start[1],end[1]+1):
                    collect.append((microstates[(k,l)].pop()))
        collections.append(collect)
    terms = {}
    for collection in collections:
        L = max(np.sum(np.array(collection)[:,::2],axis=1))
        S = max(np.sum(np.array(collection)[:,1::2],axis=1))
        if int(S) == S:
            S = int(S)
        multiplicity = int(2*S+1)
        if multiplicity in terms.keys():
            terms[multiplicity].append(terminators[L])
        else:
            terms[multiplicity] = [terminators[L]]
    return terms


def LS_terms_in_crystal_terms(group_label, l, num_electrons):
    allowed_LS_terms = LS_allowed_terms(l, num_electrons)
    Ls = range(0,l*num_electrons+1)
    reduction = {}
    for L in Ls:
        the_split = l_splitter(group_label,L)
        the_big_term = l_from_num_to_lett[L].upper()
        for split_irrep, count in the_split.dict.items():
            if split_irrep not in reduction:
                reduction[split_irrep] = []
            reduction[split_irrep].append(the_big_term)
    spin_reduction = {}
    for mult, Ls in allowed_LS_terms.items():
        for irrep, cterms in reduction.items():
            inters = [c for c in Ls if c in cterms]
            if len(inters) > 0:
                spin_reduction[(mult,irrep)] = Counter(inters)
    return spin_reduction

if __name__ == '__main__':
    print(welcome)
    Dq = sp.Symbol('Dq')
    γexps = {2: 4.42, 3: 4.50, 4: 4.61, 5: 4.48, 6: 4.81, 7: 4.63, 8: 4.71}
    Bexps =  {2: 860, 3:918 , 4:965 , 5:860, 6: 1065, 7: 971, 8:1030}
    atoms = dict(zip([2,3,4,5,6,7,8], 'V,3 Cr,3 Mn,3 Mn,2 Co,3 Co,2 Ni,2'.split(' ')))
    figsize = (17/2.54,20/2.54)
    Dqs = np.linspace(-4,4,200)
    ymax = 75
    ymin = 0
    Dqmax = np.max(Dqs)
    Dqmin = np.min(Dqs)
    if not recompute_data:
        all_all_terms = pickle.load(open(all_terms_ref_file,'rb'))
    else:
        print("Parsing CG coefficients...")
        B, C = sp.Symbol('B'), sp.Symbol('C')
        group_label = 'O'
        group = CPGs.get_group_by_label(group_label)
        group.component_labels[sp.Symbol('A_1')] = 'e_1'
        group.component_labels[sp.Symbol('A_2')] = 'e_2'
        group.component_labels[sp.Symbol('E')] = 'u v'
        group.component_labels[sp.Symbol('T_1')] = '\\alpha \\beta \\gamma'
        group.component_labels[sp.Symbol('T_2')] = '\\xi \\eta \\zeta'
        for k in group.component_labels:
            group.component_labels[k] = list(map(sp.Symbol,group.component_labels[k].split(' ')))

        rawcgs = '''
    A1xA1xA1
    1

    A1xA2xA2
    1

    A1xExE
    1 0
    0 1

    A1xT1xT1
    1 0 0
    0 1 0
    0 0 1

    A1xT2xT2
    1 0 0
    0 1 0
    0 0 1

    A2xA2xA1
    -1

    A2xExE
    0 -1
    1 0

    A2xT1xT2
    1 0 0
    0 1 0
    0 0 1

    A2xT2xT1
    -1 0 0
    0 -1 0
    0 0 -1

    ExExA1xA2xE
    1/2 0 -1/2 0
    0 1/2 0 1/2
    0 -1/2 0 1/2
    1/2 0 1/2 0

    ExT1xT1xT2
    -1/4 0 0 3/4 0 0
    0 -1/4 0 0 -3/4 0
    0 0 1 0 0 0
    3/4 0 0 1/4 0 0
    0 -3/4 0 0 1/4 0
    0 0 0 0 0 -1

    ExT2xT1xT2
    -3/4 0 0 -1/4 0 0
    0 3/4 0 0 -1/4 0
    0 0 0 0 0 1
    -1/4 0 0 3/4 0 0
    0 -1/4 0 0 -3/4 0
    0 0 1 0 0 0

    T1xT1xA1xExT1xT2
    -1/3 1/6 -1/2 0 0 0 0 0 0
    0 0 0 0 0 -1/2 0 0 -1/2
    0 0 0 0 1/2 0 0 -1/2 0
    0 0 0 0 0 1/2 0 0 -1/2
    -1/3 1/6 1/2 0 0 0 0 0 0
    0 0 0 -1/2 0 0 -1/2 0 0
    0 0 0 0 -1/2 0 0 -1/2 0
    0 0 0 1/2 0 0 -1/2 0 0
    -1/3 -4/6 0 0 0 0 0 0 0

    T1xT2xA2xExT1xT2
    -1/3 -1/2 -1/6 0 0 0 0 0 0
    0 0 0 0 0 1/2 0 0 -1/2
    0 0 0 0 1/2 0 0 1/2 0
    0 0 0 0 0 1/2 0 0 1/2
    -1/3 1/2 -1/6 0 0 0 0 0 0
    0 0 0 1/2 0 0 -1/2 0 0
    0 0 0 0 1/2 0 0 -1/2 0
    0 0 0 1/2 0 0 1/2 0 0
    -1/3 0 4/6 0 0 0 0 0 0

    T2xT2xA1xExT1xT2
    1/3 -1/6 1/2 0 0 0 0 0 0
    0 0 0 0 0 1/2 0 0 1/2
    0 0 0 0 -1/2 0 0 1/2 0
    0 0 0 0 0 -1/2 0 0 1/2
    1/3 -1/6 -1/2 0 0 0 0 0 0
    0 0 0 1/2 0 0 1/2 0 0
    0 0 0 0 1/2 0 0 1/2 0
    0 0 0 -1/2 0 0 1/2 0 0
    1/3 4/6 0 0 0 0 0 0 0
'''
        cgs = {}
        reductions = {}
        for line in rawcgs.split('\n'):
            line = line.lstrip()
            if line == '':
                continue
            if "x" in line:
                line = line.replace('1','_1').replace('2','_2')
                irrep0, irrep1, reducedto = line.split('x')[0], line.split('x')[1], line.split('x')[2:]
                reducedto = list(map(sp.Symbol, reducedto))
                key = (sp.Symbol(irrep0), sp.Symbol(irrep1))
                reductions[key] = reducedto
                cgs[key] = []
            else:
                nums = list(map(sp.S, line.split(' ')))
                nums = [sp.sign(num) * sp.sqrt(sp.sign(num)*num) for num in nums]
                cgs[key].append(nums)
        for ir0ir1key in cgs:
            cgs[ir0ir1key] = sp.Matrix(cgs[ir0ir1key])
            print(ir0ir1key)
            display(cgs[ir0ir1key])

        clebsch_gees = {}
        clebsch_gees_partitioned = {}
        for ir0ir1key in cgs:
            clebsch_gees_partitioned[ir0ir1key] = {}
            Γ0, Γ1 = ir0ir1key
            clebsch_gees_partitioned[(ir0ir1key[1], ir0ir1key[0])] = {}
            comps_0 = group.component_labels[Γ0]
            comps_1 = group.component_labels[Γ1]
            comps_2 = sum([group.component_labels[key] for key in reductions[ir0ir1key]],[])
            row = 0
            for idx0, c0 in enumerate(comps_0):
                for idx1, c1 in enumerate(comps_1):
                    # row = idx0 * len(comps_0) + idx1
                    # if ir0ir1key == (sp.Symbol('E'), sp.Symbol('T_2')):
                    #     print(row)
                    matrix_row = cgs[ir0ir1key][row,:]
                    matrix_vals = {(c0, c1, c2):val for (c2, val) in zip(comps_2, matrix_row)}
                    # if (Γ0, Γ1) == (sp.Symbol('E'), sp.Symbol('E')):
                    #     print(matrix_vals)
                    clebsch_gees.update(matrix_vals)
                    clebsch_gees_partitioned[ir0ir1key].update(matrix_vals)
                    row += 1
                    if Γ0 != Γ1:
                        matrix_vals = {(c1, c0, c2):val for  (c2, val) in zip(comps_2, matrix_row)}
                        clebsch_gees.update(matrix_vals)
                        if (Γ1, Γ0) not in clebsch_gees_partitioned:
                            clebsch_gees_partitioned[(Γ1, Γ0)] = {}
                        clebsch_gees_partitioned[(Γ1, Γ0)].update(matrix_vals)
        group.CG_coefficients = clebsch_gees
        group.CG_coefficients_partitioned = clebsch_gees_partitioned

        sym_bases = {}
        sym_bases[sp.Symbol('E')] = [
            [Qet({(2, 0) : 1}), # u
            Qet({(2, 2) : 1/sp.sqrt(2),
                (2,-2) : 1/sp.sqrt(2)})] # v
                ]

        sym_bases[sp.Symbol('T_2')] = [
                    [sp.I/sp.sqrt(2) * Qet({(2,1):1, (2,-1):1}), # xi
                    -1/sp.sqrt(2) * Qet({(2,1):1, (2,-1):-1}), # eta
                    -sp.I/sp.sqrt(2) * Qet({(2,2):1, (2,-2):-1}) # zeta
                    ]
                ]

        basis_change = {}
        for irrep, qets in sym_bases.items():
            components = group.component_labels[irrep]
            if isinstance(qets, list):
                qets = qets[0]
            for component, qet in zip(components, qets):
                basis_change[component] = qet

        orbital_basis_change = {}
        for k in basis_change:
            for spin in [S_DOWN, S_UP]:
                new_k = SpinOrbital(k,spin)
                orbital_basis_change[new_k] = Qet({k+(spin,):v for k,v in basis_change[k].dict.items()})

        def parse_headers(headers_string):
            headers_string = headers_string.replace('sup','{}^').replace(' ','')
            return tuple(list(map(sp.Symbol, headers_string.split(','))))
        
        chunk = {n:{} for n in range(2,6)}
        chunk[2] = OrderedDict({((0, sp.Symbol('A_1')), parse_headers('t_2^2,e^2')) : [[10*B + 5*C, sp.sqrt(6)*(2*B+C)],
                                [sp.sqrt(6)*(2*B + C), 8*B + 4*C]],
        ((0,sp.Symbol('E')), parse_headers('t_2^2,e^2') ): [[B + 2*C, -2*sp.sqrt(3)*B],
                            [-2*sp.sqrt(3)*B, 2*C]],
        ((0,sp.Symbol('T_2')), parse_headers('t_2^2,t_2{e}')): [[B + 2*C, 2*sp.sqrt(3)*B],
                            [2*sp.sqrt(3)*B, 2*C]],
        ((1, sp.Symbol('T_1')), parse_headers('t_2^2,t_2{e}') ): [[-5*B, 6*B],
                                [6*B, 4*B]],
        ((0, sp.Symbol('T_1')), parse_headers('t_2{e}')): [[4*B + 2*C]],
        ((1, sp.Symbol('T_2')), parse_headers('t_2{e}')): [[-8*B]],
        ((1, sp.Symbol('A_2')), parse_headers('e^2')): [[-8*B]],      
        })

        for k,v in chunk[2].items():
            chunk[2][k] = sp.Matrix(v)

        chunk[3] = OrderedDict([
        (((sp.S(1)/2, sp.Symbol('T_2')), parse_headers('t_2^3, t_2^2(sup3T_1)e, t_2^2(sup1T_2)e, t_2e^2(sup1A_1), t_2e^2(sup1E)') ) , [
            [(5*C)/sp.S(2), -3*sp.sqrt(3)*B, -5*sp.sqrt(3)*B, 4*B + 2*C, 2*B],
            [0, (-6*B+3*C)/sp.S(2), 3*B, -3*sp.sqrt(3)*B, -3*sp.sqrt(3)*B],
            [0,0, (4*B +3*C)/sp.S(2), -sp.sqrt(3)*B, sp.sqrt(3)*B],
            [0,0,0, (6*B+5*C)/sp.S(2), 10*B],
            [0,0,0,0,(-2*B+3*C)/sp.S(2)]
            ]),
        (((sp.S(1)/2,sp.Symbol('T_1')), parse_headers('t_2^3, t_2^2(sup3T_1)e, t_2^2(sup1T_2)e, t_2e^2(sup3A_2), t_2e^2(sup1E)') ), [
            [(-6*B+3*C)/sp.S(2), -3*B, 3*B, 0, -2*sp.sqrt(3)*B],
            [0, 3*C/sp.S(2), -3*B, 3*B, 3*sp.sqrt(3)*B],
            [0, 0, (-6*B+3*C)/sp.S(2), -3*B, -sp.sqrt(3)*B],
            [0,0,0, (-6*B+3*C)/sp.S(2),2*sp.sqrt(3)*B],
            [0,0,0,0, (-2*B + 3*C)/sp.S(2)]]),
        (((sp.S(1)/2,sp.Symbol('E')), parse_headers('t_2^3, t_2^2(sup1A_1)e, t_2^2(sup1E)e, e^3') ), [
            [(-6*B+3*C)/sp.S(2), -6*sp.sqrt(2)*B, -3*sp.sqrt(2)*B,0],
            [0,(8*B+6*C)/sp.S(2),10*B, sp.sqrt(3)*(2*B+C)],
            [0,0,(-B+3*C)/sp.S(2),2*sp.sqrt(3)*B],
            [0,0,0,(-8*B+4*C)/sp.S(2)]]),
        (((sp.S(3)/2,sp.Symbol('T_1')), parse_headers('t_2^2(sup3T_1)e, t_2e^2(sup3A_2)') ), [[-3*B/sp.S(2),6*B],
                                        [0,-6*B]]),
        (((sp.S(3)/2,sp.Symbol('A_2')), parse_headers('t_2^3') ), [[-15*B/sp.S(2)]]),
        (((sp.S(3)/2,sp.Symbol('T_2')), parse_headers('t_2^2(sup3T_1)e') ), [[-15*B/sp.S(2)]]),
        (((sp.S(1)/2,sp.Symbol('A_1')), parse_headers('t_2^2(sup1E)e') ), [[(-11*B+3*C)/sp.S(2)]]),
        (((sp.S(1)/2,sp.Symbol('A_2')), parse_headers('t_2^2(sup1E)e') ), [[(9*B + 3*C)/sp.S(2)]]),
        ])
        chunk[3] = OrderedDict([(k,(sp.Matrix(v)+sp.Matrix(v).T)) for k,v in chunk[3].items()])

        chunk[4] = OrderedDict([
        ((sp.S(3)/2, sp.Symbol('T_2')) , [
            [(5*C)/sp.S(2), -3*sp.sqrt(3)*B, -5*sp.sqrt(3)*B, 4*B + 2*C, 2*B],
            [0, (-6*B+3*C)/sp.S(2), 3*B, -3*sp.sqrt(3)*B, -3*sp.sqrt(3)*B],
            [0,0, (4*B +3*C)/sp.S(2), -sp.sqrt(3)*B, sp.sqrt(3)*B],
            [0,0,0, (6*B+5*C)/sp.S(2), 10*B],
            [0,0,0,0,(-2*B+3*C)/sp.S(2)]
            ]),
        ((sp.S(1)/2,sp.Symbol('T_1')), [
            [(-6*B+3*C)/sp.S(2), -3*B, 3*B, 0, -2*sp.sqrt(3)*B],
            [0, 3*C/sp.S(2), -3*B, 3*B, 3*sp.sqrt(3)*B],
            [0, 0, (-6*B+3*C)/sp.S(2), -3*B, -sp.sqrt(3)*B],
            [0,0,0, (-6*B+3*C)/sp.S(2),2*sp.sqrt(3)*B],
            [0,0,0,0, (-2*B + 3*C)/sp.S(2)]]),
        ((sp.S(1)/2,sp.Symbol('E')), [
            [(-6*B+3*C)/sp.S(2), -6*sp.sqrt(2)*B, -3*sp.sqrt(2)*B,0],
            [0,(8*B+6*C)/sp.S(2),10*B, sp.sqrt(3)*(2*B+C)],
            [0,0,(-B+3*C)/sp.S(2),2*sp.sqrt(3)*B],
            [0,0,0,(-8*B+4*C)/sp.S(2)]]),
        ((sp.S(3)/2,sp.Symbol('T_1')), [[-3*B/sp.S(2),6*B],
                                        [0,-6*B]]),
        ((sp.S(3)/2,sp.Symbol('A_2')), [[-15*B/sp.S(2)]]),
        ((sp.S(3)/2,sp.Symbol('T_2')), [[-15*B/sp.S(2)]]),
        ((sp.S(1)/2,sp.Symbol('A_1')), [[(-11*B+3*C)/sp.S(2)]]),
        ((sp.S(1)/2,sp.Symbol('A_2')), [[(9*B + 3*C)/sp.S(2)]]),
        ])

        raw_table = '''
    :3T_1, t_2^4, t_2^3({}^2T_1)e, t_2^3({}^2T_2)e, t_2^2({}^3T_1)e^2({}^1A_1), t_2^2({}^3T_1)e^2({}^1E), t_2^2({}^1T_2)e^2({}^3A_2), {t_2}{e^3}
    -15*B+5*C  s(6)*B  3*s(2)*B  -s(2)*(2*B+C)  2*s(2)*B  0  0
    0  -11*B+4*C  5*s(3)*B  s(3)*B  -s(3)*B  3*B  s(6)*B
    0  0  -3*B+6*C  -3*B  -3*B  5*s(3)*B  s(2)*(B+C)
    0  0  0  -B+6*C  -10*B  0  3*s(2)*B
    0  0  0  0  -9*B+4*C  -2*s(3)*B  -3*s(2)*B
    0  0  0  0  0  -11*B+4*C  s(6)*B
    0  0  0  0  0  0  -16*B+5*C

    :1T_2, t_2^4, t_2^3(sup2T_1)e, t_2^3(sup2T_2)e, t_2^2(sup3T_1)e^2(sup3A_2), t_2^2(sup1T_2)e^2(sup1E), t_2^2(sup1T_2)e^2(sup1A_1), {t_2}{e^3}
    -9*B+7*C  -3*s(2)*B  5*s(6)*B  0  2*s(2)*B  -s(2)*(2*B+C)  0
    0  -9*B+6*C  -5*s(3)*B  3*B  -3*B  -3*B  -s(6)*B
    0  0  3*B+8*C  -3*s(3)*B  5*s(3)*B  -5*s(3)*B  s(2)*(3*B+C)
    0  0  0  -9*B+6*C  -6*B  0  -3*s(6)*B
    0  0  0  0  -3*B+6*C  -10*B  s(6)*B
    0  0  0  0  0  5*B+8*C  s(6)*B
    0  0  0  0  0  0  7*C

    :1A_1, t_2^4, t_2^3(sup1E)e, t_2^2(sup1A_1)e^2(sup1A_1), t_2^2(sup1E)e^2(sup1E), e^4
    10*C  -12*s(2)*B  s(2)*(4*B+2*C)  2*s(2)*B  0
    0  6*C  -12*B  -6*B  0
    0  0  14*B+11*C  20*B  s(6)*(2*B+C)
    0  0  0  -3*B+6*C  2*s(6)*B
    0  0  0  0  -16*B+8*C

    :1E, t_2^4, t_2^3(sup2E)e, t_2^2(sup1E)e^2(sup1A_1), t_2^2(sup1A_1)e^2(sup1E), t_2^2(sup1E)e^2(sup1E)
    -9*B+7*C  -6*B  -s(2)*(2*B+C)  2*B  4*B
    0  -6*B+6*C  -3*s(2)*B  -12*B  0
    0  0  5*B+8*C  10*s(2)*B  -10*s(2)*B
    0  0  0  6*B+9*C  0
    0  0  0  0  -3*B+6*C

    :3T_2, t_2^3(sup2T_1)e, t_2^3(sup2T_2)e, t_2^2(sup3T_1)e^2(sup3A_2), t_2^2(sup3T_1)e^2(sup1E), t_2{e^3}
    -9*B+4*C  -5*s(3)*B  s(6)*B  s(3)*B  -s(6)*B
    0  -5*B+6*C  -3*s(2)*B  3*B  s(2)*(3*B+C)
    0  0  -13*B+4*C  -2*s(2)*B  -6*B
    0  0  0  -9*B+4*C  3*s(2)*B
    0  0  0  0  -8*B+5*C

    :1T_1, t_2^3(sup2T_1)e, t_2^3(sup2T_2)e, t_2^2(sup1T_2)e^2(sup1E), t_2{e^3}
    -3*B+6*C  5*s(3)*B  3*B  s(6)*B
    0  -3*B+8*C  -5*s(3)*B  s(2)*(B+C)
    0  0  -3*B+6*C  -s(6)*B
    0  0  0  -16*B+7*C

    :3E, t_2^3(sup4A_2)e, t_2^3(sup2E)e, t_2^2(sup1E)e^2(sup3A_2)
    -13*B+4*C  -4*B  0
    0  -10*B+4*C  -3*s(2)*B
    0  0  -11*B+4*C

    :3A_2, t_2^3(sup2E)e, t_2^2(sup1A_1)e^2(sup3A_2)
    -8*B+4*C  -12*B
    0  -2*B+7*C

    :1A_2, t_2^3(sup2E)e, t_2^2(sup1E)e^2(sup1E)
    -12*B+6*C  6*B
    0  -3*B+6*C

    :5E, t_2^3(sup4A_2)e
    -21*B

    :5T_2, t_2^2(sup3T_1)e^2(sup3A_2)
    -21*B

    :3A_1, t_2^3(sup2E)e
    -12*B+4*C
    '''
        raw_table = raw_table.replace('sup','{}^').replace('s','sqrt')
        parsed_tables = {}
        row = []
        for line in raw_table.split('\n'):
            line = line.lstrip()
            if line == '':
                continue
            if ':' in line:
                line = line.replace(' ','').replace(':','').split(',')
                term_symbol = sp.Symbol('{}^%s{%s}' % (line[0][0], line[0][1:]))
                term = ((int(line[0][0]) - 1)/sp.S(2), sp.Symbol(line[0][1:]))
                headers = tuple(list(map(sp.Symbol, line[1:])))
                matrix = []
                continue
            else:
                row = list(map(sp.S, line.split('  ')))
                matrix.append(row)
            if len(row) == len(matrix):
                matrix = sp.Matrix(matrix)
                for i in range(matrix.rows):
                    matrix[i,i] = matrix[i,i]/sp.S(2)
                matrix = matrix + matrix.T
                parsed_tables[(term, headers)] = matrix
        chunk[4] = parsed_tables

        raw_table = '''
    :2T_2, t_2^5, t_2^4(sup3T_1)e, t_2^4(sup1T_2)e, t_2^3(sup2T_1)e^2(sup3A_2), t_2^3(sup2T_1)e^2(sup1E), t_2^3(sup2T_2)e^2(sup1A_1), t_2^3(sup2T_2)e^2(sup1E), t_2^2(sup1T_2)e^3, t_2^2(sup3T_1)e^3, t_2{e^4}
    -20*B+10*C  -3*S(6)*B  -S(6)*B  0  -2*S(3)*B  4*B+2*C  2*B  0  0  0
    0  -8*B+9*C  3*B  -S(6)*B/2  3*S(2)*B/2  -3*S(6)*B/2  -3*S(6)*B/2  0  -4*B-C  0
    0  0  -18*B+9*C  -3*S(6)*B/2  3*S(2)*B/2  -5*S(6)*B/2  5*S(6)*B/2  -C  0  0
    0  0  0  -16*B+8*C  2*S(3)*B   0  0  -3*S(6)*B/2  -S(6)*B/2  0
    0  0  0  0  -12*B+8*C  -10*S(3)*B  0  3*S(2)*B/2  3*S(2)*B/2  -2*S(3)*B
    0  0  0  0  0  2*B+12*C  0  -5*S(6)*B/2  -3*S(6)*B/2  4*B+2*C
    0  0  0  0  0  0  -6*B+10*C  -5*S(6)*B/2  3*S(6)*B/2  -2*B
    0  0  0  0  0  0  0  -18*B+9*C  3*B  -S(6)*B
    0  0  0  0  0  0  0  0  -8*B+9*C  -3*S(6)*B
    0  0  0  0  0  0  0  0  0  -20*B+10*C

    :2T_1, t_2^4(sup3T_1)e, t_2^4(sup1T_2)e, t_2^3(sup2T_1)e^2(sup1A_1), t_2^3(sup2T_1)e^2(sup1E), t_2^3(sup2T_2)e^2(sup3A_2), t_2^3(sup2T_2)e^2(sup1E), t_2^2(sup1T_2)e^3, t_2^3(sup3T_1)e^3
    -22*B+9*C  -3*B  3*S(2)*B/2  -3*S(2)*B/2  3*S(2)*B/2  3*S(6)*B/2  0  -C
    0  -8*B+9*C  -3*S(2)*B/2  -3*S(2)*B/2  -15*S(2)*B/2  -5*S(6)*B/2  -4*B-C  0
    0  0  -4*B+10*C  0  0  10*S(3)*B  3*S(2)*B/2  -3*S(2)*B/2
    0  0  0  -12*B+8*C  0  0  -3*S(2)*B/2  -3*S(2)*B/2
    0  0  0  0  -10*B+10*C  2*S(3)*B  15*S(2)*B/2  -3*S(2)*B/2
    0  0  0  0  0  -6*B+10*C  5*S(6)*B/2  -3*S(6)*B/2
    0  0  0  0  0  0  -8*B+9*C  -3*B
    0  0  0  0  0  0  0  -22*B+9*C

    :2E, t_2^4(sup1A_1)e, t_2^4(sup1E)e, t_2^3(sup2E)e^2(sup1A_1), t_2^3(sup2E)e^2(sup3A_2), t_2^3(sup2E)e^2(sup1E), t_2^2(sup1E)e^3, t_2^2(sup1A_1)e^3
    -4*B+12*C  -10*B  6*B  6*S(3)*B  6*S(2)*B  -2*B  4*B+2*C
    0  -13*B+9*C  3*B  -3*S(3)*B  0  -2*B-C  -2*B
    0  0  -4*B+10*C  0  0  -3*B  -6*B
    0  0  0  -16*B+8*C  2*S(6)*B  -3*S(3)*B  6*S(3)*B
    0  0  0  0  -12*B+8*C  0  6*S(2)*B
    0  0  0  0  0  -13*B+9*C  -10*B
    0  0  0  0  0  0  -4*B+12*C

    :2A_1, t_2^4(sup1E)e, t_2^3(sup2E)e^2(sup1E), t_2^3(sup4A_2)e^2(sup3A_2), t_2^2(sup1E)e^3
    -3*B+9*C  3*S(2)*B  0  -6*B-C
    0  -12*B+8*C  -4*S(3)*B  3*S(2)*B
    0  0  -19*B+8*C  0
    0  0  0  -3*B+9*C

    :2A_2, t_2^4(sup1E)e,  t_2^3(sup2E)e^2(sup1E), t_2^2(sup1E)e^3
    -23*B+9*C  -3*S(2)*B  2*B-C
    0  -12*B+8*C  -3*S(2)*B
    0  0  -23*B+9*C

    :4T_1, t_2^4(sup3T_1)e, t_2^3(sup2T_2)e^2(sup3A_2), t_2^2(sup3T_1)e^3
    -25*B+6*C  3*S(2)*B  -C
    0  -16*B+7*C  -3*S(2)*B
    0  0  -25*B+6*C

    :4T_2, t_2^4(sup3T_1)e, t_2^3(sup2T_1)e^2(sup3A_2), t_2^2(sup3T_1)e^3
    -17*B+6*C  -S(6)*B  -4*B-C
    0  -22*B+5*C  -S(6)*B
    0  0  -17*B+6*C

    :4E, t_2^3(sup2E)e^2(sup3A_2), t_2^3(sup4A_2)e^2(sup1E)
    -22*B+5*C  -2*S(3)*B
    0  -21*B+5*C

    :6A_1, t_2^3(sup4A_2)e^2(sup3A_2)
    -35*B

    :4A_1, t_2^3(sup4A_2)e^2(sup3A_2)
    -25*B+5*C

    :4A_2, t_2^3(sup4A_2)e^2(sup1A_1)
    -13*B+7*C
    '''
        raw_table = raw_table.replace('sup','{}^').replace('S','sqrt')
        parsed_tables = {}
        for line in raw_table.split('\n'):
            line = line.lstrip()
            if line == '':
                continue
            if ':' in line:
                line = line.replace(' ','').replace(':','').split(',')
                term_symbol = sp.Symbol('{}^%s{%s}' % (line[0][0], line[0][1:]))
                term = ((int(line[0][0]) - 1)/sp.S(2), sp.Symbol(line[0][1:]))
                headers = tuple(list(map(sp.Symbol, line[1:])))
                matrix = []
            else:
                row = list(map(sp.S, line.split('  ')))
                # print(len(row))
                matrix.append(row)
            if len(row) == len(matrix):
                matrix = sp.Matrix(matrix)
                for i in range(matrix.rows):
                    matrix[i,i] = matrix[i,i]/sp.S(2)
                matrix = matrix + matrix.T
                parsed_tables[(term, headers)] = matrix
        chunk[5] = parsed_tables
        coulomb_repulsion_matrices = chunk

        tsk_matrices = coulomb_repulsion_matrices
        term_headers = {}
        for num_electrons, matrices in tsk_matrices.items():
            term_headers[num_electrons] = {}
            for term_key, matrix in matrices.items():
                term_headers[num_electrons][term_key[0]] = term_key[1]
        TSK_matrices = {}
        for num_electrons, matrices in tsk_matrices.items():
            TSK_matrices[num_electrons] = {}
            for term_key, matrix in matrices.items():
                TSK_matrices[num_electrons][term_key[0]] = matrix

        ########################################################################
        ########################################################################

        all_all_terms = {}
        verbose = False
        for num_electrons in target_electrons:
            print(num_electrons)
            n_electrons = []
            for config in config_layout('O', 2, num_electrons):
                Γ1s = config[0]
                if len(config) == 1:
                    Γ2s = []
                else:
                    Γ2s = config[1]
                n_electrons.append((config,CrystalElectronsLLCoupling('O', Γ1s, Γ2s, group)))

            # group them in terms
            all_terms = OrderedDict()
            all_qets = {}
            Λ = namedtuple('Λ',['electrons','term','γ']) 
            component_labels = group.component_labels
            wave_keys = []
            for a_config in n_electrons:
                for wave_key, wave_qet in a_config[1].equiv_waves.items():
                    term_sector = wave_key.terms[-1]
                    if Counter(wave_key.electrons) == Counter([sp.Symbol('E'),sp.Symbol('E'),sp.Symbol('T_2')]):
                        wave_keys.append(wave_key)
                    term_pivot = component_labels[term_sector[1]][0]
                    all_qets[wave_key] = wave_qet
                    if term_sector not in all_terms:
                        all_terms[term_sector] = OrderedDict()
                    if wave_key.γ == term_pivot:
                        if int(wave_key.S) == wave_key.S:
                            if wave_key.M == 0:
                                all_terms[term_sector][wave_key] = (wave_qet)
                        else:
                            if wave_key.M == S_HALF:
                                all_terms[term_sector][wave_key] = (wave_qet)

            ########################################################################
            ########################################################################

            solution = {'matrices': {},
                        'sympy expressions': {},
                        'numpy functions': {}}

            for term, term_waves in all_terms.items():
                this_waves = term_waves
                energy_matrix = []
                for idx0, qet0 in enumerate(term_waves.values()):
                    row = []
                    for idx1, qet1 in enumerate(term_waves.values()):
                        two_braket = double_electron_braket(qet0, qet1)
                        matrix_element = double_braket_basis_change(two_braket, orbital_basis_change)
                        matrix_element_as_slater = matrix_element.apply(to_slater_params)
                        mat_element = sp.expand(matrix_element_as_slater.as_symbol_sum()).subs(SLATER_TO_RACAH)
                        row.append(mat_element)
                    energy_matrix.append(row)
                energy_matrix = sp.Matrix(energy_matrix)
                energy_matrix = sp.expand(energy_matrix).subs({sp.Symbol('A'):0})
                solution['matrices'][term] = energy_matrix
                try:
                    eigenfuns = list(the_block.eigenvals().keys())
                    solution['sympy expressions'][term] = eigenfuns
                    solution['numpy functions'][term] = [sp.lambdify((sp.Symbol('Dq'),sp.Symbol('\\gamma_{CB}')), eigenfun) for eigenfun in eigenfuns]
                except:
                    solution['sympy expressions'][term] = None
                    solution['numpy functions'][term] = None

            ########################################################################
            ########################################################################

            checks = {}
            if num_electrons in [2,3,4,5]:
                for k, v in solution['matrices'].items():
                    print('='*30)
                    term_symb = sp.Symbol('{}^{%d}{%s}' % (k[0]*2+1, k[1]))
                    from_tsk = TSK_matrices[num_electrons][k]
                    if verbose:
                        display(term_symb)
                        display(v)
                        display(from_tsk)
                    random_subs = {sp.Symbol('B'): np.random.random(), sp.Symbol('C'): np.random.random()}
                    v = v.subs(random_subs)
                    from_tsk = from_tsk.subs(random_subs)
                    v_spectrum = np.sort(np.linalg.eigvalsh(np.array(v,dtype=np.float64)))
                    from_tsk_spectrum = np.sort(np.linalg.eigvalsh(np.array(from_tsk,dtype=np.float64)))
                    spectrum_diff = np.sqrt(np.sum((v_spectrum - from_tsk_spectrum)**2))
                    print("Checking spectrum equality...")
                    if (spectrum_diff < 1e-6):
                        print(">>> Check passed.", "Δ =", spectrum_diff)
                        checks[k] = True
                    else:
                        print(">>> Check failed.")
                        print(v_spectrum, from_tsk_spectrum)
                        checks[k] = False
                print('='*30)
                print("num_electrons =", num_electrons)
                for k,v in checks.items():
                    if v:
                        checkmark = '✓'
                    else:
                        checkmark = 'X'
                    print('|',checkmark,k)
                print('='*30)
                if all(checks.values()):
                    print("All checks passed.")
                else:
                    print("Not all checks passed.")
            compendium = {'checks': checks,
                        'n_electrons': n_electrons,
                        'all_terms': all_terms,
                        'solution': solution,
                        'all_qets': all_qets}
            all_all_terms[num_electrons] = compendium

        l = 2
        Dq = sp.Symbol('Dq')
        verbose = False
        for num_electrons in target_electrons:
            print(num_electrons)
            all_terms = all_all_terms[num_electrons]
            all_terms['solution']['full_matrices'] = {}
            LScterms = LS_terms_in_crystal_terms('O', l, num_electrons)
            for term_key, matrix in all_terms['solution']['matrices'].items():
                term_symbol = sp.Symbol('{}^%d{%s}' %(term_key[0]*2+1, term_key[1]))
                LSterms = LScterms[(term_key[0]*2+1, term_key[1])]
                LSterms = ','.join(sorted(LSterms.elements()))
                LSterms = '(%s)' % LSterms
                term_symbol = sp.Symbol(str(term_symbol) + LSterms)
                
                electron_counter = [Counter(psi.electrons) for psi in all_terms['all_terms'][term_key]]
                matrix_headers = [make_multi_term_symbol(psi_key) for psi_key in all_terms['all_terms'][term_key]]
                matrix_headers = sp.Matrix(matrix_headers).T
                full_matrix = sp.Matrix(sp.BlockMatrix([[matrix_headers],[matrix]]))
                row_energies = [(-4*c[(sp.Symbol('T_2'))] + 6*c[(sp.Symbol('E'))])*Dq for c in electron_counter]
                energy_term = sp.diag(*row_energies)
                full_energy_matrix = matrix + energy_term
                full_energy_matrix = full_energy_matrix.subs({sp.Symbol('C'):sp.Symbol('\\gamma_{CB}'), sp.Symbol('B'):1})
                if verbose:
                    display(term_symbol)
                    display(full_matrix)
                all_terms['solution']['full_matrices'][term_key] = full_energy_matrix
            all_all_terms[num_electrons] = all_terms

    if create_figures:
        for num_electrons in target_electrons:
            print(num_electrons)
            sol_matrices = all_all_terms[num_electrons]['solution']['full_matrices']
            γexp = γexps[num_electrons]
            Bexp = Bexps[num_electrons]
            atom = atoms[num_electrons].split(',')[0]
            charge_state =  int(atoms[num_electrons].split(',')[1])
            ion = Ion(atom, charge_state)

            exp_levels = ion.nist_data_levels['Level (eV)'] * UnitCon.con_factor('eV','cm^{-1}') / Bexp
            term_labels =  ion.nist_data_levels['Term']
            selector = exp_levels < ymax
            exp_levels = exp_levels[selector]
            term_labels = term_labels[selector]

            fig, ax = plt.subplots(figsize=figsize)
            energy_array = []
            min_energies = []
            cterm_labels = sum([['${}^%d{%s}$' % (tkey[0]*2+1, tkey[1])]*sol_matrices[tkey].rows for tkey in sol_matrices.keys()],[])
            switchboard = []
            for aDq in Dqs:
                energies = [list(np.linalg.eigvalsh(np.array( matrix.subs({sp.Symbol('\\gamma_{CB}'): γexp}).subs({Dq:aDq}),dtype=np.float64))) 
                            for matrix in sol_matrices.values()]
                all_energies = sum(energies,[])
                min_energy = min(all_energies)
                min_arg = np.argmin(np.array(all_energies))
                switchboard.append(min_arg)
                energy_array.append(sum(energies,[]))
                min_energies.append(min_energy)
            energy_array = np.array(energy_array).T
            min_energies = np.array(min_energies)
            rpivots = []
            dmin = 3
            texts = []
            tprops = dict(facecolor='w', alpha=0.3, edgecolor='w', pad=0)
            tprops2 = dict(facecolor='w', alpha=0.9, edgecolor='w', pad=0, boxstyle='round')
            lstyle = '-'
            switchjoints = [x for x in list(Dqs[(np.diff(switchboard,append=0) != 0)]) if np.abs(x)>0.2 and x not in [Dqmin, Dqmax]]
            for swj in switchjoints:
                ax.plot([swj,swj], [ymin,ymax], 'k:', alpha=0.5, lw=0.5)
                t = ax.text(swj, 10, '%.2f' % swj, rotation=90, bbox=tprops2, size=5, ha='center',va='center')
            for idx, row in enumerate(energy_array):
                men = row-min_energies
                the_label = cterm_labels[idx]
                ax.plot(Dqs[men<=ymax], men[men<=ymax], label=the_label)
                col = ax.lines[-1].get_color()
                frame_type = ('r' if (men[-1]<ymax) else 't')
                if frame_type == 'r':
                    loco = men[-1]
                    t = ax.text(Dqmax, loco, the_label, c=col, ha='center', va='center', backgroundcolor='w', bbox=tprops)
                    texts.append(t)
                    rpivots.append(loco)
                elif frame_type == 't':
                    # need to find where it hits the top part
                    loco = np.interp(ymax,men[Dqs>0],Dqs[Dqs>0])
                    if loco == Dqmin or abs(loco) < 0.1:
                        continue
                    t= ax.text(loco, ymax, the_label,c=col, ha='center', va='center', backgroundcolor='w', bbox=tprops)
                    texts.append(t)
                frame_type = ('r' if (men[0]<ymax) else 't')
                if frame_type == 'r':
                    loco = men[0]
                    # if len(rpivots)>0:
                    #     if np.min(np.abs(loco-np.array(rpivots))) < dmin:
                    #         loco = loco + dmin
                    t = ax.text(Dqmin, loco, the_label, c=col, ha='right', va='center', backgroundcolor='w', bbox=tprops)
                    texts.append(t)
                    rpivots.append(loco)
                elif frame_type == 't':
                    # need to find where it hits the top part
                    loco = np.interp(ymax,men[Dqs<0][::-1],Dqs[Dqs<0][::-1])
                    # print("-",the_label,loco)
                    if abs(loco) < 0.1:
                        continue
                    t = ax.text(loco, ymax, the_label, c=col, ha='center', va='bottom', backgroundcolor='w', bbox=tprops)
                    texts.append(t)
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', linestyle = ':', alpha=0.6))
            
            y_str = []
            first = True
            
            for exp_level, term_label in zip(exp_levels, term_labels):
                dx = -0.5
                line = plt.Line2D([-0.8+Dqmin+dx,-0.75+Dqmin+dx], [exp_level]*2, c='r')
                line.set_clip_on(False)
                ax.add_line(line)
                if (len(y_str)>0 and min(([abs(ys - exp_level) for ys in y_str])) > 2) or first:
                    plt.text(-.85+Dqmin+dx, exp_level, term_label, ha='right', va='center', c='r')
                    first = False
                y_str.append(exp_level)
            
            ax.set_ylim(ymin,ymax+4)
            ax.set_xlim(Dqmin-0.4,Dqmax+0.4)
            ax.set_xlabel('Dq/B')
            ax.set_ylabel('E/B')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            axright = ax.twinx()  # instantiate a second axes that shares the same x-axis
            axright.set_ylim(ymin*Bexp*UnitCon.con_factor('cm^{-1}','eV'), Dqmax*Bexp*UnitCon.con_factor('cm^{-1}','eV'))
            axright.set_ylabel('E/eV')
            axright.spines['top'].set_visible(False)
            axright.spines['right'].set_visible(False)
            axright.spines['bottom'].set_visible(False)
            axright.spines['left'].set_visible(False)
            ax.set_title('%s${}^{%d\!+}$\nN=%d, C/B=%.2f, B=%.0f $cm^{-1}$' % (atom, charge_state, num_electrons, γexp, Bexp))
            ax.plot([Dqmin,Dqmin,Dqmax,Dqmax],[ymin,ymax,ymax,ymin],'k:',lw=0.5)
            ax.plot([Dqmax+0.4,Dqmax+0.4],[ymin,ymax],'k:',lw=0.5, clip_on = False)
            ax.plot([0,0],[ymin,ymax],'k:',lw=0.5, clip_on = False)
            ax.plot([Dqmin-0.4,Dqmin-0.4],[ymin,ymax],'k:',lw=0.5, clip_on = False)
            plt.tight_layout()
            savefile = fig_template % num_electrons
            print("Saving figure to %s." % savefile)
            plt.savefig(savefile)
            plt.close()

    if parse_latex:
        print("Parsing LaTeX output")
        # create latex output
        # section label
        # figure with caption
        # coulomb repulsion matrices
        tex_output_lines = []
        l = 2
        for num_electrons in target_electrons:
            sol_matrices = all_all_terms[num_electrons]['solution']['matrices']
            figsize = 0.95
            if num_electrons == 2:
                figsize = 0.8
            if num_electrons != 2:
                line = '\\newpage\n\\section{$d^%d$}\n' % num_electrons
            else:
                line = '\\section{$d^%d$}\n' % num_electrons
            tex_output_lines.append(line)
            LScterms = LS_terms_in_crystal_terms('O', l, num_electrons)
            line = '''\\begin{figure}[ht!]
        \\centering
        \\includegraphics[width=%.1f\\textwidth]{./img/TSK-GT-%d.pdf}
        \\end{figure}
        \\newpage''' % (figsize, num_electrons)
            all_terms = all_all_terms[num_electrons]
            tex_output_lines.append(line)
            if num_electrons in [4,5,6]:
                tex_output_lines.append('\\begin{landscape}')
            for term_key, matrix in sol_matrices.items():
                term_symbol = sp.Symbol('{}^%d{%s}' %(term_key[0]*2+1, term_key[1]))
                LSterms = LScterms[(term_key[0]*2+1, term_key[1])]
                tempTerms = ['{}^{%d}{%s}' % (term_key[0]*2+1, tea) for tea in sorted(LSterms.elements())]
                LSterms = ','.join(tempTerms)
                LSterms = '(%s)' % LSterms
                term_symbol = sp.Symbol(str(term_symbol) + LSterms)
                electron_counter = [Counter(psi.electrons) for psi in all_terms['all_terms'][term_key]]
                matrix_headers = [make_multi_term_symbol(psi_key) for psi_key in all_terms['all_terms'][term_key]]
                matrix_headers = sp.Matrix(matrix_headers).T
                row_energies = [(-4*c[(sp.Symbol('T_2'))] + 6*c[(sp.Symbol('E'))])*Dq for c in electron_counter]
                energy_term = sp.diag(*row_energies)
                full_energy_matrix = matrix + energy_term
                full_matrix = sp.Matrix(sp.BlockMatrix([[matrix_headers],[full_energy_matrix]]))
                matrix_latex = '''\\begin{equation}
        \\tiny
        %s
        \\end{equation}\n''' % sp.latex(full_matrix)
                term_line = '\\centerline{$%s$}\n' % sp.latex(term_symbol)
                tex_output_lines.append(term_line)
                tex_output_lines.append(matrix_latex)
            if num_electrons in [4,5,6]:
                tex_output_lines.append('\\end{landscape}')
        tex_output = '\n'.join(tex_output_lines)
        print("Saving latex output to %s" % output_latex_file)
        open(output_latex_file,'w').write(tex_output)
