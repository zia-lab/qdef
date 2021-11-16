#!/usr/bin/env python

######################################################################
#                          __     ____                               #
#               ____ _____/ /__  / __/________  ________             #
#              / __ `/ __  / _ \/ /_/ ___/ __ \/ ___/ _ \            #
#             / /_/ / /_/ /  __/ __/ /__/ /_/ / /  /  __/            #
#             \__, /\__,_/\___/_/  \___/\____/_/   \___/             #
#               /_/                                                  #
#                                                                    #
######################################################################


import pickle
import sympy as sp
import os
import numpy as np
import pandas as pd
from sympy.physics.quantum import Ket, Bra
from sympy.physics.wigner import gaunt
from collections import OrderedDict
from itertools import product

module_dir = os.path.dirname(__file__)


# =============================================================== #
# ====================== Load element data ====================== #

name_to_symb = pickle.load(open(os.path.join(module_dir,'data',
                                            'name_to_symb.pkl'),'rb'))
name_to_num  = pickle.load(open(os.path.join(module_dir,'data',
                                            'name_to_num.pkl'),'rb'))
symb_to_name  = pickle.load(open(os.path.join(module_dir,'data',
                                            'symb_to_name.pkl'),'rb'))
symb_to_num  = pickle.load(open(os.path.join(module_dir,'data',
                                            'symb_to_num.pkl'),'rb'))
num_to_name  = pickle.load(open(os.path.join(module_dir,'data',
                                            'num_to_name.pkl'),'rb'))
num_to_symb  = pickle.load(open(os.path.join(module_dir,'data',
                                            'num_to_symb.pkl'),'rb'))

atomicGoodies  = pickle.load(open(os.path.join(module_dir,'data',
                                            'atomicGoodies.pkl'),'rb'))
ionization_data  = pickle.load(open(os.path.join(module_dir,'data',
                                            'ionization_data.pkl'),'rb'))['data']

atom_symbs   = list(symb_to_name.keys())
atom_names   = list(name_to_num.keys())
GT_CGs = pickle.load(open(os.path.join(module_dir,'data',
                                            'GT_CG.pkl'),'rb'))
CG_coeffs_partitioned = pickle.load(open(os.path.join(module_dir,'data',
                                            'CG_coeffs_partitioned.pkl'),'rb'))

nistdf = pd.read_pickle(os.path.join(module_dir,'data',
                                    'nist_atomic_spectra_database_levels.pkl'))
spinData = pd.read_pickle(os.path.join(module_dir,'data','spindata.pkl'))

# gives the atomic numbers of the firs three rows of transition
# metals, the keys correspond to the row of periodic table
# to which they correspond
element_groups = {
'transition metals': {4: list(range(21,31)),
                      5: list(range(39,49)),
                      6: list(range(71,80))}}

# ====================== Load element data ====================== #
# =============================================================== #

# =============================================================== #
# ===================== Load group theory data ================== #

group_dict = pickle.load(open(os.path.join(module_dir,'data','gtpackdata.pkl'),'rb'))
vcoeffs = pickle.load(open(os.path.join(module_dir,'data','Vcoeffs.pkl'),'rb'))
group_data = group_dict['group_data']
metadata = group_dict['metadata']

# ===================== Load group theory data ================== #
# =============================================================== #

def orthogonal_matrix(euler_params):
    '''
    For a given symmetry operation as parametrized by Euler angles
    α, β, γ, and by its  determinant  det (±1).  The corresponding
    orthogonal matrix is returned.
    '''
    α, β, γ, det = euler_params
    row_0 = [-sp.sin(α)*sp.sin(γ) + sp.cos(α)*sp.cos(β)*sp.cos(γ),
             -sp.sin(α)*sp.cos(γ) - sp.sin(γ)*sp.cos(α)*sp.cos(β),
              sp.sin(β)*sp.cos(α)]
    row_1 = [sp.sin(α)*sp.cos(β)*sp.cos(γ) + sp.sin(γ)*sp.cos(α),
            -sp.sin(α)*sp.sin(γ)*sp.cos(β) + sp.cos(α)*sp.cos(γ),
            sp.sin(α)*sp.sin(β)]
    row_2 = [-sp.sin(β)*sp.cos(γ),
             sp.sin(β)*sp.sin(γ),
             sp.cos(β)]
    mat = det*sp.Matrix([row_0,row_1,row_2])
    return mat

class HartreeFockData():
    '''
    Repo of data from the land of Hartree-Fock.
    '''
    HFradavg = pickle.load(open(os.path.join(module_dir,'data','HFravgs.pkl'),'rb'))
    HFsizes = pickle.load(open(os.path.join(module_dir,'data','HFsizes.pkl'),'rb'))
    ArabicToRoman = dict(zip(range(1,36),[
                    'I','II','III','IV','V','VI','VII','VIII','IX','X',
                    'XI',  'XII',  'XIII',  'XIV', 'XV', 'XVI', 'XVII',
                    'XVIII',      'XIX',     'XX','XXI','XXII','XXIII',
                    'XXIV', 'XXV','XXVI','XXVII','XXVIII','XXIX','XXX',
                    'XXXI','XXXII','XXXIII','XXXIV','XXXV']
                    )
                    )
    num_to_symb  = num_to_symb
    @classmethod
    def radial_average(cls, element, charge_state, n):
        '''
        Returns  the radial average <r^n> for a valence electron for
        the  given element and charge state (n=0 neutral, n=1 singly
        ionized, ...) within the limitations of Hartree-Fock.

        The  element  can be given either as its atomic number or by
        its symbol.

        Data is taken from Fraga's et al Handbook of Atomic Data.

        The unit for the provided radial average is Angstrom^n.

        Provided data has 5 significant figures.
        '''
        charge_state = int(charge_state)
        assert charge_state >= 0, "What odd ion state you speak of?"
        charge_state = cls.ArabicToRoman[charge_state+1]
        if isinstance(element, int):
            element = cls.num_to_symb[element]
        try:
            return float(cls.HFradavg['<r^%d>' % n].loc[[element]][charge_state])
        except:
            raise ValueError('This radial average is not here.')
    @classmethod
    def atom_size(cls, element, charge_state):
        '''
        Size of given element with given charge.
        Given in Angstrom.
        '''
        if isinstance(element, int):
            element = cls.num_to_symb[element]
        charge_state = cls.ArabicToRoman[charge_state+1]
        return float(cls.HFsizes.loc[[element]][charge_state])

class Atom():
    '''
    From these everything is made up.
    '''
    def __init__(self, kernel):
        '''
        Object can be initialized either by giving an atomic number,
        element name, or element symbol.
        '''
        if kernel in range(1, 119):
            self.atomic_number = kernel
            self.symbol = num_to_symb[kernel]
            self.name = num_to_name[kernel]
        elif (kernel in atom_names) or (kernel.lower() in atom_names):
            self.name = kernel.lower()
            self.symbol = name_to_symb[self.name]
            self.atomic_number = name_to_num[self.name]
        elif kernel in atom_symbs:
            self.symbol = kernel
            self.atomic_number = symb_to_num[kernel]
            self.name = symb_to_name[kernel]
        else:
            raise ValueError('to initialize input must be either an atomic' +
                             ' number, element name, or element symbol')
        # a dictionary with known ionization energies at different stages
        if self.symbol in ['Db','Sg','Bh','Hs','Mt','Ds','Rg',
                           'Cn','Nh','Fl','Mc','Lv','Ts','Og']:
            self.ionization_energies = []
        else:
            self.ionization_energies = ionization_data[self.symbol]
        # a dataframe with level data as compiled by NIST
        self.nist_data = nistdf[nistdf['Element'] == self.symbol]
        # a dataframe with isotopic data
        self.isotope_data = spinData[spinData['atomic_number'] == self.atomic_number]
        # additional data
        # the electronegativity of the free neutral atom
        self.electronegativity = atomicGoodies["Electronegativity"][self.atomic_number]
        # a latex string for the electron configuration of the ground state
        self.electron_configuration = atomicGoodies["Electron Config"][self.atomic_number]
        # crystal structure of its most common solid form
        self.common_crystal_structure = atomicGoodies["Crystal Structure"][self.atomic_number]
        # electron configuration
        self.electron_configuration_string = atomicGoodies["Electron Config String"][self.atomic_number]
        # Van der Waals radius
        self.van_der_waals_radius =  atomicGoodies["Van der Waals radius"][self.atomic_number]
        # Atomic radius
        self.atomic_radius = atomicGoodies["Atomic Radius"][self.atomic_number]
        # Covalent radius
        self.covalent_radius = atomicGoodies["Covalent Radius"][self.atomic_number]
        # Block
        self.block = atomicGoodies["Block"][self.atomic_number]
        # Period
        self.period = atomicGoodies["Period"][self.atomic_number]
        # Series
        self.series = atomicGoodies["Series"][self.atomic_number]
        # Group
        self.group = atomicGoodies["Group"][self.atomic_number]

    def level_diagram(self, charge, min_energy=-np.inf, max_energy=np.inf):
        '''make a nice plot of the levels of the ion with the given charge'''
        cmap = plt.cm.RdYlGn
        datum = self.nist_data[self.nist_data['Charge'] == charge]
        energy_levels = datum['Level (eV)']
        configs = datum['Configuration']
        if charge == 0:
            fig_name = '%s' % (self.symbol)
            latex_name = fig_name
        else:
            fig_name = '%s +%d' % (self.symbol, charge)
            latex_name = '{%s}^{+%d}' % (self.symbol, charge)
        plt.close(fig_name)
        fig, ax = plt.subplots(figsize=(2, 5), num=fig_name)
        annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        for level_energy in energy_levels:
            if ((level_energy < max_energy) & (level_energy > min_energy)):
                ax.plot([0, 1], [level_energy]*2, 'k-', lw=0.5)
        level_anchors = ax.scatter([0]*len(datum['Level (eV)']),
                                   datum['Level (eV)'],
                                   c='k', s=4)
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('E / eV')
        ax.set_title('$%s$' % latex_name)
        plt.tight_layout()

        def update_annot(ind):
            pos = level_anchors.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            indices = ind["ind"]
            text = (configs[indices[0]] + '\n' +
                    ('%.3f eV' % (energy_levels[indices[0]])))
            annot.set_text(text)
            annot.get_bbox_patch().set_facecolor('white')

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = level_anchors.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()

    def __repr__(self):
        return '%s : %s : %d' % (self.name, self.symbol, self.atomic_number)

    def __str__(self):
        return '%s : %s : %d' % (self.name, self.symbol, self.atomic_number)

class Ion(Atom):
    '''
    Same as an Atom, but with the added attribute of charge_state.
    Also, the spectroscopic data is limited to that ion.
    '''
    def __init__(self, element, charge_state):
        Atom.__init__(self,element)
        self.charge_state = charge_state
        self.nist_data = self.nist_data[self.nist_data['Charge'] == self.charge_state]
        self.nist_data.reset_index(drop=True, inplace=True)

class PeriodicTable():
    '''
    It basically instantiates all atoms.
    '''
    pt_group_symbols = {1: ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
                     2: ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
                     3: ['Sc', 'Y', 'Lu', 'Lr'],
                     4: ['Ti', 'Zr', 'Hf', 'Rf'],
                     5: ['V', 'Nb', 'Ta', 'Db'],
                     6: ['Cr', 'Mo', 'W', 'Sg'],
                     7: ['Mn', 'Tc', 'Re', 'Bh'],
                     8: ['Fe', 'Ru', 'Os', 'Hs'],
                     9: ['Co', 'Rh', 'Ir', 'Mt'],
                     10: ['Ni', 'Pd', 'Pt', 'Ds'],
                     11: ['Cu', 'Ag', 'Au', 'Rg'],
                     12: ['Zn', 'Cd', 'Hg', 'Cn'],
                     13: ['B', 'Al', 'Ga', 'In', 'Tl', 'Nh'],
                     14: ['C', 'Si', 'Ge', 'Sn', 'Pb', 'Fl'],
                     15: ['N', 'P', 'As', 'Sb', 'Bi', 'Mc'],
                     16: ['O', 'S', 'Se', 'Te', 'Po', 'Lv'],
                     17: ['F', 'Cl', 'Br', 'I', 'At', 'Ts'],
                     18: ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og'],
                     'Lanthanides': ['La','Ce','Pr','Nd','Pm','Sm',
                            'Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb'],
                     'Actinides': ['Ac','Th','Pa','U','Np','Pu','Am',
                            'Cm','Bk','Cf','Es','Fm','Md','No']}
    transition_metals = (
             {4: ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'],
              5: ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd'],
              6: ['Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'],
              7: ['Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']})
    def __init__(self):
        self.atoms = {i:Atom(i) for i in range(1,119)}
    def annotated_ptable(self, fig, annotations, line_color = 'white', accent_color = 'red'):
        ax = fig.add_subplot(1,1,1)
        margin = 0.02
        font_multiplier = 2.2
        num_annotations = True
        if isinstance(list(annotations.keys())[0],str):
            num_annotations = False
        for atom in self.atoms.values():
            if atom.group != 'f-block':
                group_loc = int(atom.group)
                row = -atom.period
            else:
                if (57 <= atom.atomic_number <= 70):
                    group_loc = atom.atomic_number - 57+3
                    row = -9+0.5
                else:
                    group_loc = atom.atomic_number - 89+3
                    row = -10+0.5
            color = line_color
            if not num_annotations:
                if atom.symbol in annotations.keys():
                    color = accent_color
                    plt.text(group_loc+0.5, row-0.8, annotations[atom.symbol],\
                    ha='center', va='center', fontsize=4*font_multiplier,\
                    color=color)
            else:
                if atom.atomic_number in annotations.keys():
                    color = accent_color
                    plt.text(group_loc+0.5, row-0.8,\
                            annotations[atom.atomic_number], ha='center',\
                            va='center', fontsize=4*font_multiplier)
            plt.text(group_loc+0.5, row-0.375, atom.symbol, ha='center',\
                            va='top', fontsize=7*font_multiplier,\
                            weight='bold', color = color)
            plt.text(group_loc+0.5, row-margin-0.1, atom.atomic_number,\
                            ha='center', va='top', fontsize=4*font_multiplier)
            plt.plot([group_loc+margin, group_loc+1-margin, group_loc+1-margin,\
                                        group_loc+margin, group_loc+margin],
                    [row-margin, row-margin, row-1+margin, row-1+margin,\
                    row-margin], '-',color=line_color,lw=0.5)
        ax.set_xlim(0.5,19.5)
        ax.set_ylim(-11.5,-0.5)
        ax.axis('off')
        ax.set_aspect('equal')
        return fig, ax

class Term():
    '''
    To  represent  a  term,  this  object  holds the states that
    correspond  to  it,  and offers a few methods in to view the
    enclosed wave functions.
    '''
    def __init__(self, init_dict):
        for k,v in init_dict.items():
            setattr(self, k, v)
        self.term_prototype = sp.Symbol(r'{{}}^{{{M}}}{ir}')
        self.state_label_prototype = sp.Symbol(r'\Psi({α}, {S}, {{{Γ}}}, {M_s}, {γ})')
    def term_symbol(self):
        return sp.Symbol(str(self.term_prototype).format(M=str(2*self.S+1),ir=str(sp.latex(self.irrep))))
    def make_state_symbols(self):
        state_symbols = []
        for qet, state_key in zip(self.states, self.state_keys):
            (Γ1, Γ2, Γ3, γ3, S, mSz) = state_key
            ket = qet.as_ket(fold_keys=True)
            ket_symbol = sp.latex(ket).replace('\\right\\rangle','\\right|')
            α = Γ1*Γ2
            term_symb = self.term_symbol()
            state_symbol = '\\Psi(%s,%s,M\!=\!%d,%s)' % (sp.latex(α).lower(), term_symb, mSz, sp.latex(γ3))
            state_symbols.append((state_symbol, ket_symbol))
        return state_symbols
    def __str__(self):
        return (self.term_symbol())
    def __repr__(self):
        return '%s: %d states' % (str(self.term_symbol()), len(self.states))

class Qet():
    '''
    A Qet is a dictionary of keys and values. Keys correspond to
    tuples   of  quantum  numbers  or  symbols  and  the  values
    correspond to the accompanying coefficients.

    Scalars may be added by using an empty tuple as a key.

    A  qet  may be multiplied by a scalar, in which case all the
    coefficients are multiplied by it. It may also be multiplied
    by   another   qet,   in  which  case  quantum  numbers  are
    concatenated and coefficients multiplied accordingly.
    '''
    def __init__(self, bits=0):
        if bits == 0:
                self.dict = {}
        elif isinstance(bits, dict):
                self.dict = {k: v for k, v in bits.items() if v!=0}

    def __add__(self, other):
        new_dict = dict(self.dict)
        if other == 0:
                return self
        for key, coeff in other.dict.items():
            if key in new_dict.keys():
                new_dict[key] += coeff
            else:
                new_dict[key] = coeff
        return Qet(new_dict)

    def __sub__(self, other):
        if other == 0:
            return self
        return Qet(dict(self.dict)) + (-1)*Qet(dict(other.dict))

    def __neg__(self):
        return (-1)*Qet(self.dict)

    def vec_in_basis(self, basis):
        '''
        Given an ordered basis  return  a  list  with  the
        coefficients of the qet in that basis.
        '''
        coeffs = [0]*len(basis)
        for key, val in self.dict.items():
            coeffs[basis.index(key)] = val
        return coeffs

    def subs(self, subs_dict):
        '''
        The substitutions in subs_dict are evaluated on the
        coeffients of the qet.
        Equal to valsubs but kept for backwards compatibility.
        '''
        new_dict = dict()
        for key, val in self.dict.items():
            new_dict[key] = sp.S(val).subs(subs_dict)
        return Qet(new_dict)

    def valsubs(self, subs_dict):
        '''
        The substitutions in subs_dict are evaluated on the
        coeffients of the qet.
        '''
        new_dict = dict()
        for key, val in self.dict.items():
            new_dict[key] = sp.S(val).subs(subs_dict)
        return Qet(new_dict)

    def keysubs(self, subs_dict):
        '''
        The substitutions in subs_dict are evaluated on the
        keys of the qet.
        It assumes that they keys are already symby symbols.
        If not, then they are converted to them.
        '''
        new_dict = dict()
        for key, val in self.dict.items():
            new_dict[sp.Symbol(key).subs(subs_dict)] = sp.S(val)
        return Qet(new_dict)

    def __mul__(self, multiplier):
        '''
        Give a representation of the qet  as  a  Ket  from
        sympy.physics.quantum, fold_keys  =  True  removes
        unnecessary parentheses and nice_negatives =  True
        assumes all numeric  keys  and  presents  negative
        values with a bar on top.
        '''
        if isinstance(multiplier, Qet):
            new_dict = dict()
            for k1, v1 in self.dict.items():
                for k2, v2 in multiplier.dict.items():
                    k3 = k1 + k2
                    v3 = v1 * v2
                    if v3 !=0:
                        new_dict[k3] = v3
            return Qet(new_dict)
        else:
            new_dict = dict(self.dict)
            for key, coeff in new_dict.items():
                new_dict[key] = multiplier*(coeff)
            return Qet(new_dict)

    def __rmul__(self, multiplier):
        '''this is required to enable multiplication
        from the left and from the right'''
        new_dict = dict()
        for key, coeff in self.dict.items():
            new_dict[key] = multiplier*(coeff)
        return Qet(new_dict)

    def basis(self):
        '''return a list with all the keys in the qet'''
        return list(self.dict.keys())

    def dual(self):
        '''conjugate all the coefficients'''
        new_dict = dict(self.dict)
        for key, coeff in new_dict.items():
            new_dict[key] = sp.conjugate(coeff)
        return Qet(new_dict)

    def as_operator(self, opfun):
        OP = sp.S(0)
        for key, val in self.dict.items():
                OP += sp.S(val) * opfun(*key)
        return OP

    def as_ket(self, fold_keys=False, nice_negatives=False):
        '''
        Give a representation of the qet  as  a  Ket  from
        sympy.physics.quantum, fold_keys  =  True  removes
        unnecessary parentheses and nice_negatives =  True
        assumes all numeric  keys  and  presents  negative
        values with a bar on top.
        '''
        sympyRep = sp.S(0)
        for key, coeff in self.dict.items():
            if key == ():
                sympyRep += coeff
            else:
                if fold_keys:
                    if nice_negatives:
                        key = tuple(sp.latex(k) if k>=0 else (r'\bar{%s}'\
                                                % sp.latex(-k)) for k in key)
                    sympyRep += coeff*Ket(*key)
                else:
                    sympyRep += coeff*Ket(key)
        return sympyRep

    def as_bra(self):
        '''
        Give a representation of the qet  as  a  Bra  from
        sympy.physics.quantum.
        '''
        sympyRep = sp.S(0)
        for key, coeff in self.dict.items():
            if key == ():
                sympyRep += coeff
            else:
                sympyRep += coeff*Bra(*key)
        return sympyRep

    def as_braket(self):
        '''
        Give a representation of the qet as a Bra*Ket. The
        keys in the dict for the ket are assumed to  split
        first half for the bra, and other second half  for
        the ket.
        '''
        sympyRep = sp.S(0)
        for key, coeff in self.dict.items():
            l = int(len(key)/2)
            if key == ():
                sympyRep += coeff
            else:
                sympyRep += coeff*(Bra(*key[:l])*Ket(*key[l:]))
        return sympyRep

    def as_symbol_sum(self):
        '''
        Take the keys multiply them by their corresponding
        coefficients,  add  them  all  up,  and return the
        resulting sympy expression.
        '''
        tot = sp.S(0)
        for k, v in self.dict.items():
            tot += v*k
        return tot

    def as_c_number_with_fun(self):
        '''
        This method can be used to apply a function  to  a
        qet.  The  provided  function  f  must   take   as
        arguments a single pair  of  qnum  and  coeff  and
        return a dictionary or a (qnum, coeff) tuple.
        '''
        sympyRep = sp.S(0)
        for key, op_and_coeff in self.dict.items():
            ops_and_coeffs = list(zip(op_and_coeff[::2],op_and_coeff[1::2]))
            for op, coeff in ops_and_coeffs:
                if key == ():
                    sympyRep += coeff
                else:
                    sympyRep += coeff*op(*key)
        return sympyRep

    def apply(self,f):
        '''
        This method can be used to apply a function  to  a
        qet the provided function f must take as arguments
        a single pair of  qnum  and  coeff  and  return  a
        dictionary or a (qnum, coeff) tuple
        '''
        new_dict = dict()
        for key, coeff in self.dict.items():
            appfun = f(key,coeff)
            if isinstance(appfun, dict):
                for key2, coeff2 in appfun.items():
                    if coeff2 != 0:
                        if key2 not in new_dict.keys():
                            new_dict[key2] = coeff2
                        else:
                            new_dict[key2] += coeff2
            else:
                new_key, new_coeff = appfun
                if new_coeff !=0:
                    if new_key not in new_dict.keys():
                        new_dict[new_key] = (new_coeff)
                    else:
                        new_dict[new_key] += (new_coeff)
        return Qet(new_dict)

    def norm(self):
        '''compute the norm of the qet'''
        norm2 = 0
        for key, coeff in self.dict.items():
            norm2 += abs(coeff)**2
        return sp.sqrt(norm2)

    def symmetrize(self):
        '''
        Use  if  the  keys  of the kets are tuples and one
        wants  to  make equal keys that are the reverse of
        one another, i.e.

        {(1,0):a, (0,1):b} -> {(1,0):a+b}
        '''
        new_dict = dict()
        for key, coeff in self.dict.items():
            rkey = key[::-1]
            if rkey in new_dict.keys():
                new_dict[rkey] += coeff
            else:
                if key in new_dict.keys():
                    new_dict[key] += coeff
                else:
                    new_dict[key] = coeff
        return Qet(new_dict)

    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return 'Qet(%s)' % str(self.dict)

symmetry_bases = pickle.load(open(os.path.join(module_dir,'data',
                                            'symmetry_bases.pkl'),'rb'))

class ProductTable():
    '''
    This class used to hold the data of a direct product of two
    irreducible representations turned  into a  direct  sum  of
    irreducible representations.
    '''
    def __init__(self, odict, irrep_labels, grp_label):
        self.odict = odict
        self.irrep_labels = irrep_labels
        self.grp_label = grp_label
    def pretty_parse(self):
        '''creates a nice latex representation of the product table'''
        irep_symbols = self.irrep_labels
        list_o_lists = [[self.odict[(ir0,ir1)] for ir0 in self.irrep_labels]\
                                                for ir1 in self.irrep_labels]
        rows = [[sp.Symbol(self.grp_label)]+irep_symbols]
        for idx, arow in enumerate(list_o_lists):
            row = [irep_symbols[idx]]
            row.extend(arow)
            rows.append(row)
        return fmt_table(rows).replace('+',r'{\oplus}')

class CrystalGroup():
    '''
    Class for a point symmetry group.
    '''
    def __init__(self, group_data_dict):
        self.index = group_data_dict['index']
        self.label = group_data_dict['group label']
        self.classes = group_data_dict['classes']
        self.irrep_labels = group_data_dict['irrep labels']
        self.character_table = group_data_dict['character table']
        self.character_table_inverse = sp.simplify(group_data_dict['character table'].T**(-1))
        self.class_labels = group_data_dict['class labels']
        self.irrep_matrices = group_data_dict['irrep matrices']
        self.generators = group_data_dict['generators']
        self.multiplication_table = group_data_dict['multiplication table']
        self.euler_angles = group_data_dict['euler angles']
        self.group_operations = group_data_dict['group operations']
        self.gen_multiplication_table_dict()
        self.order = len(self.group_operations)
        self.operations_matrices = {k: orthogonal_matrix(v) for k, v in self.euler_angles.items()}
        self.irrep_dims = {k: list(v.values())[0].shape[0] for k, v in self.irrep_matrices.items()}
        self.direct_product_table()
        self.component_labels = self.get_component_labels()
        self.symmetry_adapted_bases = symmetry_bases[self.label]
        self.CG_coefficients = GT_CGs[self.label]
        self.CG_coefficients_partitioned = CG_coeffs_partitioned[self.label]
        self.gen_char_table_dict()
        if self.label in vcoeffs.keys():
            self.V_coefficients = vcoeffs[self.label]
        else:
            self.V_coefficients = None

    def gen_char_table_dict(self):
            self.character_table_dict = {irrep_label: \
                    {class_label: self.character_table[
                                  self.irrep_labels.index(irrep_label), \
                                  self.class_labels.index(class_label)] \
                                      for class_label in self.class_labels} \
                                      for irrep_label in self.irrep_labels}

    def gen_multiplication_table_dict(self):
            multiplication_table_dict = {}
            group_ops = self.group_operations
            for op0, op1 in product(group_ops, group_ops):
                row_idx = group_ops.index(op0)
                col_idx = group_ops.index(op1)
                op0op1 = self.multiplication_table[row_idx, col_idx]
                multiplication_table_dict[(op0,op1)] = op0op1
            self.multiplication_table_dict = multiplication_table_dict

    def get_component_labels(self):
        '''
        Generate the labels for the components of all the group
        irreducible representations.
        Labeling is done based on the size of the corresponding
        irreducible representation.
        1D -> a_{irrep label}
        2D -> u_{irrep label}, v_{irrep label}
        3D -> x_{irrep label}, y_{irrep label}, z_{irrep label}
        '''
        irrep_dims = self.irrep_dims
        components = {}
        for irrep_label, irrep_dim in irrep_dims.items():
            str_label = str(irrep_label)
            c_labels = []
            if irrep_dim == 1:
                c_labels = [sp.Symbol('a_{%s}' % str_label)]
            elif irrep_dim == 2:
                c_labels = [sp.Symbol('u_{%s}' % str_label),
                            sp.Symbol('v_{%s}' % str_label)]
            elif irrep_dim == 3:
                c_labels = [sp.Symbol('x_{%s}' % str_label),
                            sp.Symbol('y_{%s}' % str_label),
                            sp.Symbol('z_{%s}' % str_label)]
            assert len(c_labels) != 0
            components[irrep_label] = c_labels
        return components

    def direct_product(self, ir0, ir1):
        '''
        Given the label for a cpg and  labels for  two
        of its irreducible  representations, determine
        the direct sum decomposition of their product.
        This product  is  returned  as a qet with keys
        corresponding  to  the irreps and values equal
        to the integer coefficients.
        '''
        # grab group classes, irrep names, and character table
        group_classes = self.classes
        group_irreps = self.irrep_labels
        group_chartable = self.character_table
        assert ir0 in group_irreps, 'irrep not in %s' % str(group_irreps)
        assert ir1 in group_irreps, 'irrep not in %s' % str(group_irreps)
        chars_0, chars_1 = [group_chartable.row(group_irreps.index(ir)) for ir in [ir0, ir1]]
        chars = sp.Matrix([char0*char1 for char0, char1 in zip(chars_0, chars_1)])
        partition = (self.character_table_inverse*chars)
        qet = Qet()
        for element, ir in zip(partition, group_irreps):
            el = int(sp.N(element,1,chop=True))
            qet = qet + Qet({ir:el})
        return qet.basis()

    def direct_product_table(self):
        '''
        This creates the complete set of  binary
        products of irreducible  representations
        for the given group.
        The result is saved as  an attribute  in
        the group.
        '''
        if hasattr(self, 'product_table'):
            return self.product_table
        group_classes = self.classes
        group_irreps = self.irrep_labels
        product_table = OrderedDict()
        for ir0 in group_irreps:
            for ir1 in group_irreps:
                if (ir1,ir0) in product_table.keys():
                    product_table[(ir0,ir1)] = product_table[(ir1,ir0)]
                else:
                    product_table[(ir0,ir1)] = self.direct_product(ir0, ir1)#.as_symbol_sum()
        self.product_table = ProductTable(product_table, group_irreps, self.label)
        return self.product_table

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label

class CPGroups():
    '''
    Class to hold all crystallographic point groups.
    '''
    def __init__(self, groups):
        self.all_group_labels = [
            'C_{1}', 'C_{i}', 'C_{2}', 'C_{s}',
            'C_{2h}', 'D_{2}', 'C_{2v}', 'D_{2h}',
            'C_{4}', 'S_{4}', 'C_{4h}', 'D_{4}',
            'C_{4v}', 'D_{2d}', 'D_{4h}', 'C_{3}',
            'S_{6}', 'D_{3}', 'C_{3v}', 'D_{3d}',
            'C_{6}', 'C_{3h}', 'C_{6h}', 'D_{6}',
            'C_{6v}', 'D_{3h}', 'D_{6h}', 'T',
            'T_{h}', 'O', 'T_{d}', 'O_{h}']
        self.groups = {k: CrystalGroup(v) for k, v in groups.items()}
        self.metadata = metadata
    def get_group_by_label(self, label):
        if '{' not in label:
            if len(label) > 1:
                label = '%s_{%s}' % label.split('_')
        group_idx = 1 + self.all_group_labels.index(label)
        return self.groups[group_idx]

class CrystalField():
    def __init__(self, group_num):
        self.group_num = group_num
        # In the groups with the largest symmetry
        # the number of free parameters can be reduced
        # and this gives a set of possible simplifications.
        # This list contains all the possible parametric forms of the crystal field.
        self.cflist = crystal_fields[group_num]
        self.simplified_ham = self.to_expression()
    def matrix_rep_symb(self, l):
        '''
        Calculates   the  matrix  representations  of  the
        crystal  field operator in the subspace of angular
        momentum l.

        The ordered basis for this representation is

        :math:`\{|l,-l\rangle,|l,-1+1\rangle,\ldots,\l,l-1\rangle,|l,l\rangle\}`

        so  that  the  top  left  element in the resulting
        matrices correspond to

        :math:`\langle l, -l | V_{CF} | l, -l\rangle`,

        and the bottom right element to

        :math:`\langle l, +l | V_{CF} | l, +l\rangle`.
        '''
        if isinstance(l,str):
            l = {'s':0,'p':1,'d':2,'f':3,'g':4}[l]
        self.symb_matrix_reps = []
        for one_cf in self.cflist:
            mat = []
            mls = list(range(-l,l+1))
            for m1 in mls:
                row = []
                for m2 in mls:
                    total = sum([sp.sqrt(sp.S(4)*sp.pi/sp.S(2*k[0]+sp.S(1)))* \
                    threeHarmonicIntegral(l,m1,k[0],k[1],l,m2)*v \
                     for (k,v) in one_cf.dict.items()])
                    row.append(total)
                mat.append(row)
            self.symb_matrix_reps.append(sp.Matrix(mat))
        if len(self.symb_matrix_reps) == 2:
            if self.symb_matrix_reps[0] == self.symb_matrix_reps[1]:
                self.symb_matrix_reps = [self.symb_matrix_reps[0]]
        return self.symb_matrix_reps
    def to_expression(self):
        return [Bsimple(sum([sp.Symbol('C_{%d,%d}' % k) * v for k,v \
                    in field.dict.items()])) for field in self.cflist]
    def splitter(self,l):
        try:
            reps = self.symb_matrix_reps
        except AttributeError:
            self.matrix_rep_symb(l)
            reps = self.symb_matrix_reps
        splits = []
        for rep in reps:
            eigen_stuff = rep.eigenvects()
            splits.append(eigen_stuff)
        return splits

class Bnm():
    '''
    A class for Bnm crystal field parameters.

    Instantiated  with  a  dictionary that must contain at least
    the following keys:

    host            (str) e.g. 'MgO'
    site            (str) e.g. 'Mg'
    point_sym_group (str) e.g. 'C_{3v}'
    ion         (str,int) e.g. ('Cr',3)
    params         (dict) e.g. {(2,0): 1.34e3, (4,2): -2.34e3}
    unit            (str) e.g. 'cm^{-1}'
    sources        (list) e.g. ['Morrison 1982, Crystal F ...', ...]
    comments       (list) e.g. ['This is very approximate.', 'Yup.']
    experimental   (bool) e.g. False

    Ideally  params should be accompanied by params_uncertainty,
    a  dictionary with the same keys than params but with values
    equal to corresponding uncertainties.
    '''
    def __init__(self, Bnm_params):
        self.min_keys = set(['host','site','point_sym_group','params',
                             'unit','sources', 'ion', 'is_experimental'])
        self.energy_units = ['cm^{-1}','eV','J']
        self.init_dict = Bnm_params
        assert self.min_keys <= set(Bnm_params.keys()), \
            'provide at least: %s' % str(self.min_keys)
        for attr_name in Bnm_params:
            setattr(self, attr_name, Bnm_params[attr_name])
        assert self.unit in self.energy_units, \
                "unit must be one of %s" % str(self.energy_units)
    def __str__(self):
        return '%s\n%s^%d+:%s\n%s' % (self.point_sym_group, self.ion[0],
                                    self.ion[1], self.host, str(self.params))
    def __repr__(self):
        return str(self.init_dict)

class Anm():
    '''
    A  class  for  Anm  crystal  field parameters. Initialized by giving a
    dictionary  whose  keys  are  3-tuples  (n(int),m(int),t(str))  with t
    either  'r'  (real),  'i' (imaginary), 'n' (norm) and whose values are
    dictionaries with keys that must be a subset of Anm.AnmGoodKeys.

    For this given dictionary of dictionaries, if the inner dictionary has
    'total' as one key, then self.totalAnm is simply a dictionary keyed by
    2-tuples  (n,m)  and  valued by the 'total' value; if 'total' is not a
    key  then  all of the inner dictionaries are added up as a function of
    (n,m).  At  this  stage  whether  the  quantity refered to is real, or
    imaginary  has  been  included  as  the  third  element of the 3-tuple
    (n,m,t),  but when the total is computed then this goes along with the
    value.

    The 'unit' key in the params dictionaries is used to infer the unit of
    length being used, and the unit of energy being considered.

    --------
    Examples

    params = {(2,0,r): {'mono': 1.34e-6, 'dipole': 1.25e-5, 'unit': 'cm^{-1}/Å^{2}'},
              (4,4,i): {'mono': 4.55e-6, 'dipole': -1.23e-6, 'unit': 'cm^{-1}/Å^{4}'}}}
    Anm({'params': params})

    '''
    Anmkeys = [('params',dict), # a dictionary of dictionaries
               ('refs',list)]
    AnmGoodKeys = ['mono','self-i','dipole','total', 'unit'] # this are the allowed keys for a specific Anm
    def __init__(self, Anmdict):
        for k in self.Anmkeys:
            attr_name, attr_type = k
            if attr_name in Anmdict:
                assert isinstance(Anmdict[attr_name],k[1]), '%s key must be of %s or not given' % (attr_name, attr_type)
                setattr(self, attr_name, Anmdict[attr_name])
            else:
                setattr(self, attr_name, None)
        if self.params:
            self.totalAnm = {}
            for key, value_dict in self.params.items():
                assert set(list(value_dict)).issubset(self.AnmGoodKeys)
                n, m, t = key
                unit = value_dict.pop('unit')
                if 'total' in value_dict.keys():
                    self.totalAnm[(n,m)] = value_dict['total']
                else:
                    self.totalAnm[(n,m)] = sum(list(value_dict.values()))
                if t == 'i':
                    self.totalAnm[(n,m)] = self.totalAnm[(n,m)]*1j
                elif t in['r', 'n']:
                    pass
            self.totalAnm_length_unit = unit.split('/')[-1].split('^')[0]
            self.totalAnm_energy_unit = unit.split('/')[0]
    def Bnm_compute(self, ion):
        '''
        This  function  collects  the  relevant  Hartree-Fock radial
        averages  and  approximates  the corresponding Bnm using the
        totalAnm parameters.

        It returns a Bnm object.
        '''
        Bnmdict = {}
        Bnmparams = {}
        assert self.totalAnm_length_unit == 'Å', 'Hartree Fock data requires unit of lenght to be Å.'
        for k,v in self.totalAnm.items():
            Bnmparams[k] = v * HartreeFockData().radial_average(ion.atomic_number,ion.charge_state,k[0])
        Bnmdict['params'] = Bnmparams
        Bnmdict['unit'] = self.totalAnm_energy_unit
        return Bnm(Bnmdict)
