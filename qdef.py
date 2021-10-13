#!/usr/bin/env python

import os, re, pickle
import numpy as np
from math import ceil 

import sympy as sp
import pandas as pd
import math
from sympy import pi, I
from sympy.physics.quantum import Ket, Bra
from sympy.physics.wigner import gaunt

from collections import OrderedDict
from itertools import product
from tqdm.notebook import tqdm

from matplotlib import pyplot as plt

from IPython.display import display, HTML, Math

from misc import *


module_dir = os.path.dirname(__file__)

# =============================================================== #
# ===================== Load group theory data ================== #

group_dict = pickle.load(open(os.path.join(module_dir,'data','gtpackdata.pkl'),'rb'))
group_data = group_dict['group_data']
metadata = group_dict['metadata']
vcoeffs_fname = os.path.join(module_dir,'data','Vcoeffs.pkl')

# ===================== Load group theory data ================== #
# =============================================================== #

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
# =========================== Load others ======================= #

morrison_loc = os.path.join(module_dir,'data','morrison.pkl')
morrison = pickle.load(open(morrison_loc,'rb'))

# =========================== Load others ======================= #
# =============================================================== #


# =============================================================== #
# =======================  Ynm eval tweak ======================= #

# To avoid simplification of negative m values, the eval method
# on the spherical  harmonics  Ynm  needs  to be redefined. All
# that is done is  commenting   out  a  portion of the original
# source code.

@classmethod
def new_eval(cls, n, m, theta, phi):
    n, m, theta, phi = [sp.sympify(x) for x in (n, m, theta, phi)]
    # Handle negative index m and arguments theta, phi
    #if m.could_extract_minus_sign():
    #    m = -m
    #    return S.NegativeOne**m * exp(-2*I*m*phi) * Ynm(n, m, theta, phi)
    if theta.could_extract_minus_sign():
        theta = -theta
        return sp.Ynm(n, m, theta, phi)
    if phi.could_extract_minus_sign():
        phi = -phi
        return sp.exp(-2*I*m*phi) * sp.Ynm(n, m, theta, phi)
sp.Ynm.eval = new_eval

# =======================  Ynm eval tweak ======================= #
# =============================================================== #

# =============================================================== #
# =========================== Classes =========================== #

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
    def __init__(self):
        self.atoms = {i:Atom(i) for i in range(1,119)}

class Qet():
    '''
    A Qet is a dictionary of  keys  and  values.  Keys
    correspond to tuples of quantum numbers or symbols
    and the  values  correspond  to  the  accompanying
    coefficients.
    Scalars may be added by using an empty tuple as  a
    key.
    A qet may be multiplied by a scalar, in which case
    all the coefficients are multiplied by it.
    It may also be multiplied by another qet, in which
    case  quantum   numbers   are   concatenated   and
    coefficients multiplied accordingly.
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
                        key = tuple(sp.latex(k) if k>=0 else (r'\bar{%s}' % sp.latex(-k)) for k in key)
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
        list_o_lists = [[self.odict[(ir0,ir1)] for ir0 in self.irrep_labels] for ir1 in self.irrep_labels]
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
        self.order = len(self.group_operations)
        self.operations_matrices = {k: orthogonal_matrix(v) for k, v in self.euler_angles.items()}
        self.irrep_dims = {k: list(v.values())[0].shape[0] for k, v in self.irrep_matrices.items()}
        self.direct_product_table()
        self.component_labels = self.get_component_labels()

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

# this is an ugly way of loading this
# but it's necessary given that having saved them
# as a pickle with included Qets fails to load given
# that unpickling needs knowing the class it's trying to load
# therefore I turned them into regular dictionaries
# and here they're converted to qets again

crystal_fields_raw = morrison['crystal_fields_raw']
crystal_fields = {}
for k in crystal_fields_raw:
    crystal_fields[k] = [Qet(q) for q in crystal_fields_raw[k]]

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
        assert self.min_keys <= set(params.keys()), \
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

# =========================== Classes =========================== #
# =============================================================== #

###########################################################################
#################### Calculation of Surface Harmonics #####################

def SubSupSymbol(radix,ll,mm):
    '''
    Generates a symbol placeholder for the B
    coefficients   in   the   crystal  field
    potential.
    '''
    SubSupSym = sp.symbols(r'{%s}_{%s}^{%s}' % (radix, str(ll), str(mm)))
    return SubSupSym

def SubSubSymbol(radix,ll,mm):
    '''
    Generates   a   symbol   placeholder   for  the  B
    coefficients in the crystal field potential.
    '''
    SubSubSym = sp.symbols(r'{%s}_{{%s}{%s}}' % (radix, str(ll), str(mm)))
    return SubSubSym

def kronecker(i,j):
    return 0 if i!=j else 1

def Wigner_d(l, n, m, beta):
    k_min = max([0,m-n])
    k_max = min([l-n,l+m])
    Wig_d_prefact = sp.sqrt((sp.factorial(l+n)
                          *sp.factorial(l+m)
                          *sp.factorial(l-n)
                          *sp.factorial(l-m)))
    Wig_d_summands = [((-sp.S(1))**(k - m + n)
                      * sp.cos(beta/2)**(2*l+m-n-2*k)
                      * sp.sin(beta/2)**(2*k+n-m)
                      / sp.factorial(l - n -k)
                      / sp.factorial(l + m - k)
                      / sp.factorial(k)
                      / sp.factorial(k-m+n)
                      )
                      for k in range(k_min,k_max+1)]
    Wig_d = (Wig_d_prefact*sum(Wig_d_summands)).doit()
    return Wig_d

def Wigner_D(l, n, m, alpha, beta, gamma):
    args = (l, n, m, alpha, beta, gamma)
    if args in Wigner_D.values.keys():
      return Wigner_D.values[args]
    if beta == 0:
      Wig_D = sp.exp(-I*m*alpha-I*m*gamma) * kronecker(n,m)
      if n == m:
        Wig_D = (sp.cos(-m*alpha-m*gamma)+I*sp.sin(-m*alpha-m*gamma))
      else:
        Wig_D = 0
    elif beta == pi:
      if n == -m:
        Wig_D = (-1)**l * (sp.cos(-m*alpha + m*gamma)+I*sp.sin(-m*alpha + m*gamma))
      else:
        Wig_D = 0
    else:
      Wig_D_0 = I**(abs(n)+n-abs(m)-m)
      Wig_D_1 = (sp.cos(-n*gamma-m*alpha) \
                 + I*sp.sin(-n*gamma-m*alpha)) * Wigner_d(l,n,m,beta)
      Wig_D = Wig_D_0 * Wig_D_1
      Wig_D = Wig_D
    return Wig_D
Wigner_D.values = {}

def real_or_imagined(qet):
    '''
    For  a given superposition of spherical harmonics,
    determine  if  the total has a pure imaginary (i),
    pure  real (r), or mixed character (m), it assumes
    that the coefficients in the superposition are all
    real.
    '''
    chunks = dict(qet.dict)
    valences = []
    for key in list(chunks.keys()):
        if key not in chunks.keys():
            continue
        l, m = key
        chunk = chunks[key]
        if (l,-m) in chunks:
            partner = chunks[(l,-m)]
            if abs(partner) == abs(chunk):
                if sp.sign(partner) == sp.sign(chunk):
                    if m%2 == 0:
                        valences.append("r")
                    else:
                        valences.append("i")
                else:
                    if m%2 == 0:
                        valences.append("i")
                    else:
                        valences.append("r")
            else:
                valences.append("m")
            chunks.pop((l,-m))
        else:
            valences.append("m")
        if m!=0: # if equal to zero this would have been done already
            chunks.pop(key)
    valences = list(set(valences))
    if len(valences) > 1:
        return "m"
    else:
        return valences[0]

def RYlm(l, m, alpha, beta, gamma, detRot):
    '''
    This  would  be  rotateHarmonic in the Mathematica
    code.   It  is  used  in  the  projection  of  the
    spherical  harmonics  to  create  symmetry adapted
    wavefunctions.
    '''
    Rf = Qet()
    for nn in range(-l,l+1):
        wigD = Wigner_D(l, m, nn, alpha, beta, gamma)
        if wigD != 0:
          Rf = Rf + Qet({(l,nn): wigD})
    return (sp.S(detRot)**l) * Rf

def flatten_matrix(mah):
    '''
    A  convenience  function to flatten a sympy matrix
    into a list of lists
    '''
    return [item for sublist in mah.tolist() for item in sublist]

def SymmetryAdaptedWF(group, l, m):
    '''
    This  returns  the  proyection  of  Y_l^m  on  the
    trivial  irreducible  representation  of the given
    group.
    '''
    if isinstance(group,str):
      group = CPGs.get_group_by_label(group)
    degree = 1
    # Order of the group which  is  equal  to
    # the number of the elements
    order = len(group.group_operations)
    SALC = Qet()
    # This sum is over all elements of the group
    for group_params in group.euler_angles.values():
        alpha, beta, gamma, detRot = group_params
        SALC = SALC + RYlm(l,m,alpha,beta,gamma,detRot)
    SALC = (sp.S(1)/order)*SALC
    SALC = SALC.apply(lambda x,y : (x, sp.simplify(y)))
    return SALC

def linearly_independent(vecs):
    '''
    Given  a  list  of vectors return the largest subset which of linearly
    independent  ones  and  the  indices  that  correspond  to them in the
    original list.
    '''
    matrix = sp.Matrix(vecs).T
    good_ones = matrix.rref()[-1]
    return good_ones, [vecs[idx] for idx in good_ones]

def SymmetryAdaptedWFs(group, l, normalize=True, verbose=False, sympathize=True):
    '''
    For  a  given  group  and  a  given  value of l, this returns a set of
    linearly  independent  symmetry adapted functions which are also real-
    valued.

    If  the set that is found initially contains combinations that are not
    purely  imaginary  or pure real, then the assumption is made that this
    set contains single spherical harmonics, and then sums and differences
    between m and -m are given by doing this through the values of |m| for
    the functions with mixed character.

    The  output is a list of dictionaries whose keys are (l,m) tuples, and
    whose values are the corresponding coefficients.
    '''
    # apply the projection operator on the trivial irreducible rep
    # and collect the resulting basis functions
    # together with the values of (l,m) included
    flags = []
    WFs = []
    complete_basis = []
    for m in range(-l,l+1):
        aWF = SymmetryAdaptedWF(group, l, m)
        if len(aWF.dict)>0:
            WFs.append(aWF)
            complete_basis.extend(aWF.basis())

    complete_basis = list(sorted(list(set(complete_basis))))
    # to see if they are linearly independent
    # convert the WFs to vectors on the basis collected
    # above
    vecs = [WF.vec_in_basis(complete_basis) for WF in WFs]
    lin_indep_idx, lin_indep_vecs = linearly_independent(vecs)

    # reduce the WFs to a linearly independent set

    WFs = [WFs[i] for i in lin_indep_idx]
    # test to see if the included WFs are real, imaginary, or mixed
    # if real, keep as is
    # if purely imaginary, multiply by I
    # if mixed then collect for further processing
    realWFs = []
    mixedWFs = []
    for WF in WFs:
        valence = real_or_imagined(WF)
        if normalize:
            norm = WF.norm()
            WF = WF*(sp.S(1)/norm)
        if valence == 'r':
            realWFs.append(WF)
        elif valence == 'i':
            realWFs.append(I*WF)
        elif valence == 'm':
            flags.append('m')
            mixedWFs.append(WF)
    # collect the values of |m| included in the mixed combos
    mixedMs = set()
    if (len(mixedWFs) != 0) and verbose:
        print("\nMixtures found, unmixing...")
    for WF in mixedWFs:
        # ASSUMPTION: both m and -m are in there and only as singles
        assert len(WF.dict) == 1
        for key, val in WF.dict.items():
            mixedMs.add(abs(key[1]))
    # for the values of m in mixedMs compute the real sums
    # and differences
    for m in mixedMs:
        if m%2 == 0:
            qp = Qet({(l,m): 1}) + Qet({(l,-m): 1})
            qm = Qet({(l,m): I}) + Qet({(l,-m): -I})
            if normalize:
                qp = qp*(sp.S(1)/sp.sqrt(2))
                qm = qm*(sp.S(1)/sp.sqrt(2))
            realWFs.append(qp)
            realWFs.append(qm)
        elif m%2 == 1:
            qp = Qet({(l,m): I}) + Qet({(l,-m): I})
            qm = Qet({(l,m): 1}) + Qet({(l,-m): -1})
            if normalize:
                qp = qp*(sp.S(1)/sp.sqrt(2))
                qm = qm*(sp.S(1)/sp.sqrt(2))
            realWFs.append(qp)
            realWFs.append(qm)
    # the resulting list of realWFs must be of equal lenght
    # than WFs which in turn is equal to the number of linearly
    # independent projectd basis functions
    if len(realWFs) != len(WFs):
        raise Exception("FAILED: there are less real combos than originally")
    # in addition
    # must check that the resulting basis is still linearly independent
    # must run through the same business of collecting all the represented
    # spherical harmonics, converting that to coefficient vectors
    # and testing for linear independence
    complete_basis = []
    for WF in realWFs:
        complete_basis.extend(WF.basis())
    complete_basis = list(sorted(list(set(complete_basis))))

    vecs = [WF.vec_in_basis(complete_basis) for WF in realWFs]
    lin_indep_idx, lin_indep_vecs = linearly_independent(vecs)
    if len(lin_indep_idx) != len(WFs):
        raise Excepction("FAILED: +- mixture was not faithful")
    # make the linearly independent vectors orthonormal
    lin_indep_vecs = list(map(list,sp.GramSchmidt([sp.Matrix(vec) for vec in lin_indep_vecs], normalize)))
    finalWFs = []
    if sympathize:
        better_vecs = []
        for vec in lin_indep_vecs:
            clear_elements = [abs(v) for v in vec if v!=0]
            if len(list(set(clear_elements))) == 1:
                better_vec = [0 if vl == 0 else sp.sign(vl) for vl in vec]
                better_vecs.append(better_vec)
            else:
                better_vecs.append(vec)
        lin_indep_vecs = better_vecs
    for vec in lin_indep_vecs:
        qdict = {k:v for k,v in zip(complete_basis, vec)}
        finalWFs.append(Qet(qdict))
    return finalWFs

generic_cf = Qet({(k,q):(sp.Symbol('B_{%d,%d}^%s' % (k,q,"r"))-sp.I*sp.Symbol('B_{%d,%d}^%s' % (k,q,"i"))) for k in [1,2,3,4,5,6] for q in range(-k,k+1)})

def compute_crystal_field(group_num):
    '''
    This  function returns a list with the possible forms that the crystal
    field  has for the given group. This list has only one element up till
    group  27,  after that the list has two possibilities that express the
    possible sign relationships between the B4q and the B6q coefficients.

    For groups 1-3 an empty list is returned.

    The  crystal  field  is  a  qet  which has as keys tuples (k,q) and as
    values sympy symbols for the corresponding coefficients.
    '''
    full_params = morrison['Bkq grid from tables 8.1-8.3 in_Morrison 1988']
    if group_num < 3:
        print("Too little symmetry, returning empty list.")
        return []
    if group_num <=27:
        cf = [generic_cf.subs(morrison['special_reps'][group_num]).subs({k:0 for k,v in full_params[group_num].items() if not v})]
    elif group_num == 28:
        cf_p = Qet({
                 (3,2): sp.Symbol('B_{3,2}^r'),
                 (3,-2): sp.Symbol('B_{3,2}^r'),
                 (4,0): sp.Symbol('B_{4,0}^r'),
                 (4,4): sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (4,-4): sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (6,0): sp.Symbol('B_{6,0}^r'),
                 (6,4): -sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 (6,-4): -sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 })
        cf_m = Qet({
                 (3,2): sp.Symbol('B_{3,2}^r'),
                 (3,-2): sp.Symbol('B_{3,2}^r'),
                 (4,0): sp.Symbol('B_{4,0}^r'),
                 (4,4): -sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (4,-4): -sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (6,0): sp.Symbol('B_{6,0}^r'),
                 (6,4): sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 (6,-4): sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 })
        cf = [cf_p, cf_m]
    elif group_num == 29:
        cf_p = Qet({
                 (4,0): sp.Symbol('B_{4,0}^r'),
                 (4,4): sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (4,-4): sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (6,0): sp.Symbol('B_{6,0}^r'),
                 (6,4): -sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 (6,-4): -sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 })
        cf_m = Qet({
                 (4,0): sp.Symbol('B_{4,0}^r'),
                 (4,4): -sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (4,-4): -sp.sqrt(sp.S(5)/sp.S(14))*sp.Symbol('B_{4,0}^r'),
                 (6,0): sp.Symbol('B_{6,0}^r'),
                 (6,4): sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 (6,-4): sp.sqrt(sp.S(7)/sp.S(2))*sp.Symbol('B_{6,0}^r'),
                 })
        cf = [cf_p, cf_m]
    elif group_num == 30:
        cf = compute_crystal_field(29)
    elif group_num == 31:
        cf = compute_crystal_field(28)
    elif group_num == 32:
        cf = compute_crystal_field(30)
    return cf

#################### Calculation of Surface Harmonics #####################
###########################################################################

###########################################################################
############### Calculation of Clebsch-Gordan Coefficients ################

def cg_symbol(comp_1, comp_2, irep_3, comp_3):
    '''
    Given  symbols  for three components (comp_1, comp_2, comp_3) of three
    irreducible  representations  of  a  group  this  function  returns  a
    sp.Symbol for the corresponding Clebsch-Gordan coefficient:

    <comp_1,comp_2|irep_3, comp_3>

    The symbol of the third irrep is given explicitly as irep_3. The other
    two  irreps are implicit in (comp_1) and (comp_2) and should be one of
    the symbols in group.irrep_labels.

    (comp_1,   comp_2,   and   comp_3)  may  be  taken  as  elements  from
    group.component_labels.
    '''
    symb_args = (comp_1, comp_2, irep_3, comp_3)
    return sp.Symbol(r"{\langle}%s,%s|%s,%s{\rangle}" % symb_args)

class V_coefficients():
    '''
    This class loads data for the V coefficients for the octahedral group
    as defined in Appendix C of Griffith's book "The  Irreducible  Tensor
    Method for Molecular Symmetry Groups".
    In here the labels for the components for the irreducible  reps  have
    been matched in the following way:
    A_1 : \iota -> a_{A_1}
    A_2 : \iota -> a_{A_2}
    E   : \theta -> u_{E} \epsilon -> v_{E}
    T_1 : x -> x_{T_1} y -> y_{T_1} z -> z_{T_1}
    T_2 : x -> x_{T_2} y -> y_{T_1} z -> z_{T_1}
    '''
    def __init__(self):
        self.coeffs = pickle.load(open(vcoeffs_fname,'rb'))
    def eval(self, args):
        '''
        Args must be a tuple of sympy symbols for irreducible representations
        and components. This  tuple  must  contain  interleaved  symbols  for
        irreducible representations and components.
        args must match the template (a,α,b,β,c,γ) that matches with
        Griffith's notation like so:
                        V ⎛ a b c ⎞
                          ⎝ α β γ ⎠.
        '''
        return self.coeffs[args]

def group_clebsch_gordan_coeffs(group, Γ1, Γ2, rep_rules = True, verbose=False):
    '''
    Given   a  group  and  symbol  labels  for  two  irreducible
    representations  Γ1  and  Γ2  this  function  calculates the
    Clebsh-Gordan  coefficients used to span the basis functions
    of  their  product  in terms of the basis functions of their
    factors.

    By  assuming  an  even phase convention for all coefficients
    the result is also given for the exchanged order (Γ2, Γ1).

    If  rep_rules  = False, this function returns a tuple with 3
    elements,  the  first  element being a matrix of symbols for
    the  CGs  coefficients  for  (Γ1, Γ2) the second element the
    matrix  for  symbols  for (Γ2, Γ1) and the third one being a
    matrix  to which its elements are matched element by element
    to the first and second matrices of symbols.

    If rep_rules = True, this function returns two dictionaries.
    The  keys  in the first one equal CGs coefficients from (Γ1,
    Γ2)  and  the second one those for (Γ2, Γ1); with the values
    being the co- rresponding coefficients.

    These  CG  symbols  are  made  according  to  the  following
    template:
        <i1,i2|i3,i4>
        (i1 -> symbol for basis function in Γ1 or Γ2)
        (i2 -> symbol for basis function in Γ2 or Γ1)
        (i3 -> symbol for an ir rep Γ3 in the group)
        (i4 -> symbol for a basis function of Γ3)
    '''
    irreps = group.irrep_labels
    irep1, irep2 = Γ1, Γ2
    # must first find the resulting direct sum decomposition of their product
    irep3s = group.product_table.odict[(irep1, irep2)]
    if irep3s == []: # this is needed in case theres a single term in the direct sum
        irep3s = [group.product_table.odict[(irep1, irep2)]]
    # also need to grab the labels for a set of generators
    generators = group.generators
    component_labels = group.component_labels
    print("Grabbing the labels for the basis functions ...") if verbose else None
    labels_1, labels_2 = component_labels[irep1], component_labels[irep2]
    cg_size = len(labels_1)*len(labels_2)
    print("CG is a ({size},{size}) matrix ...".format(size=cg_size)) if verbose else None
    generators_1 = [group.irrep_matrices[irep1][g] for g in generators]
    generators_2 = [group.irrep_matrices[irep2][g] for g in generators]

    # then create the list of linear constraints
    print("Creating the set of linear constraints ...") if verbose else None
    # In (2.31) there are five quantities that determine one constraints
    all_eqns = []
    for irep3 in irep3s:
        labels_3 = component_labels[irep3]
        for generator in generators:
            D1, D2, D3 = [group.irrep_matrices[irep][generator] for irep in [irep1,irep2,irep3]]
            γ1s, γ2s, γ3s = [list(range(D.shape[0])) for D in [D1,D2,D3]]
            for γ1, γ2, γ3p in product(γ1s, γ2s, γ3s):
                lhs = []
                for γ1p in γ1s:
                    for γ2p in γ2s:
                        symb_args = (labels_1[γ1p],labels_2[γ2p],irep3,labels_3[γ3p])
                        chevron = cg_symbol(*symb_args)
                        coeff = D1[γ1, γ1p]*D2[γ2,γ2p]
                        if coeff:
                            lhs.append(coeff*chevron)
                lhs = sum(lhs)
                rhs = []
                for γ3 in γ3s:
                    symb_args = (labels_1[γ1],labels_2[γ2],irep3,labels_3[γ3])
                    chevron = cg_symbol(*symb_args)
                    coeff = D3[γ3, γ3p]
                    if coeff:
                        rhs.append(coeff*chevron)
                rhs = sum(rhs)
                eqn = rhs-lhs
                if (eqn not in all_eqns) and (-eqn not in all_eqns) and (eqn != 0):
                    all_eqns.append(eqn)

    # collect all the symbols included in all_eqns
    free_symbols = set()
    for eqn in all_eqns:
        free_symbols.update(eqn.free_symbols)
    free_symbols = list(free_symbols)

    # convert to matrix of coefficients
    coef_matrix = sp.Matrix([[eqn.coeff(cg) for cg in free_symbols] for eqn in all_eqns])
    # and simplify using the rref
    rref = coef_matrix.rref()[0]
    # turn back to symbolic and solve
    better_eqns = [r for r in rref*sp.Matrix(free_symbols) if r!=0]
    # return better_eqns, free_symbols
    try:
        better_sol = sp.solve(better_eqns, free_symbols)
    except:
        better_sol = sp.solve(better_eqns, free_symbols, manual=True)
    # construct the unitary matrix with all the CGs
    print("Building the big bad matrix ...") if verbose else None
    U_0 = []
    U_1 = []
    for γ1 in labels_1:
        for γ2 in labels_2:
            row_0 = []
            row_1 = []
            for irep3 in irep3s:
                labels_3 = component_labels[irep3]
                for γ3 in labels_3:
                    # the given order and the exchanged one
                    # is saved here to take care of the phase
                    # convention upon exchange of Γ1 and Γ2
                    chevron_0 = cg_symbol(γ1, γ2, irep3, γ3)
                    chevron_1 = cg_symbol(γ2, γ1, irep3, γ3)
                    row_0.append(chevron_0)
                    row_1.append(chevron_1)
            U_0.append(row_0)
            U_1.append(row_1)
    Usymbols_0 = sp.Matrix(U_0)
    Usymbols_1 = sp.Matrix(U_1)
    # replace with the solution to the linear constraints
    U_0 = Usymbols_0.subs(better_sol)
    # build the unitary constraints
    print("Bulding the unitarity constraints and assuming U to be orthogonal ...") if verbose else None
    unitary_constraints = U_0*U_0.T - sp.eye(U_0.shape[0])
    # flatten and pick the nontrivial ones
    unitary_set = [f for f in flatten_matrix(unitary_constraints) if f!=0]
    # solve
    try:
        unitary_sol = sp.solve(unitary_set)
    except:
        unitary_sol = sp.solve(unitary_set, manual=True)
    print("%d solutions found ..." % len(unitary_sol)) if verbose else None
    Usols = [U_0.subs(sol) for sol in unitary_sol]
    sol_0 = Usols[0]
    if not rep_rules:
        return Usymbols_0, Usymbols_1, sol_0
    else:
        dic_0 = {k:v for k,v in zip(list(Usymbols_0), list(sol_0))}
        dic_1 = {k:v for k,v in zip(list(Usymbols_1), list(sol_0))}
        return dic_0, dic_1

############### Calculation of Clebsch-Gordan Coefficients ################
###########################################################################

###########################################################################
################################## Others #################################

def mbasis(l):
    '''
    Return a row matrix with symbols corresponding to the kets
    that span the angular momentum space for a given value of l.
    '''
    return sp.Matrix([[sp.Symbol('|%d,%d\\rangle' % (l,ml)) for ml in range(-l,l+1)]])

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


def threeHarmonicIntegral(l1, m1, l2, m2, l3, m3):
    '''
    Returns the value of the three spherical harmonics integral,
    the variety with the first one having a complex conjugate.

    - It may be non-zero only if l1 + l2 + l2 is even.
    - It is non-zero unless |l1-l2| <= l3 <= l1+l2

    .. math:: \int d\Omega Y_{l_1,m_1}^* Y_{l_2,m_2} Y_{l_3,m_3}

    '''
    return sp.S(-1)**m1 * gaunt(l1,l2,l3,-m1,m2,m3)


def l_splitter(group_num_or_label, l):
    '''
    This  function  takes  a  value of l and determines how many
    times  each  of the irreducible representations of the given
    group  is contained in the reducible representation obtained
    from   the   irreducible  representation  of  the  continous
    rotation group as represented by the set of Y_{l,m}.

    More simply stated it returns how states that transform like
    an  l=2  would split into states that would transform as the
    irreducible representations of the given group.

    The   function  returns  a  Qet  whose  keys  correspond  to
    irreducible  representation  symbols  of the given group and
    whose  values  are  how many times they are contained in the
    reducible representation of the group.

    ----------
    Parameters

    group_num_or_label  :  int  or  str  Index  or  string for a
    crystallographic  point  group.  l  :  int  or  str azimutal
    quantum number (arbitrary) or string character s,p,d,f,g,h,i

    -------
    Examples

    A  d-electron  in  an  cubic  field  splits into states that
    transform  either  like  E or like T_2. This can be computed
    with this function like so:

    l_splitter('O', 2) -> Qet({E: 1, T_2: 1})

    An if one were interested in how the states of an f-electron
    split under C_4v symmetry, one could

    l_splitter('C_{4v}',  3)  -> Qet({A_2: 1, B_2: 1, B_1: 1, E:
    2})

    '''
    if isinstance(l, str):
        l = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}[l]
    l = sp.S(l)
    if isinstance(group_num_or_label, str):
        group_num = CPGs.all_group_labels.index(group_num_or_label)+1
    else:
        group_num = group_num_or_label
    group = CPGs.groups[group_num]
    charVec = []
    # calculate the character of each of the classes
    # in the reducible representation for the given l
    # this requires iterating over all the classes
    # and picking up any of its symmetry operations
    # figuring out what is the angle of rotation
    # that corresponds to the rotation part of
    # the matrix that represents it
    for group_class in group.classes:
        group_op = group.classes[group_class][0]
        _, _, _, detRot = group.euler_angles[group_op]
        rot_matrix = group.operations_matrices[group_op]*detRot
        cos_eta = (rot_matrix.trace()-1)/sp.S(2)
        eta = sp.acos(cos_eta)
        if sp.N(eta,chop=True) == 0:
            char = 2*l + 1
        else:
            char = sp.sin((l+sp.S(1)/2)*eta)/sp.sin(eta/2)
        charVec.append(char)
    # Once the characters have been computed
    # the inverse of the matrix representing the character
    # table for the group can be used.
    charVec = sp.N(sp.Matrix(charVec),chop=True)
    splitted = list(map(lambda x: round(sp.N(x,chop=True)),
                list(group.character_table_inverse*charVec)))
    return Qet(dict(zip(group.irrep_labels,splitted)))

def Bsimple(Bexpr):
    '''
    Takes   a   sympy   expression,   finds   the  Bnm
    coefficients  in  it and if there's only a real or
    an  imaginary  part then it returns the expression
    without the ^r or ^i.
    '''
    free_symbs = list(Bexpr.free_symbols)
    symb_counter = {}
    for symb in free_symbs:
        if 'B_' not in str(symb):
            continue
        kqcombo = eval(str(symb).split('{')[-1].split('}')[0])
        reorim = str(symb).split('^')[-1]
        if kqcombo not in symb_counter.keys():
            symb_counter[kqcombo] = 1
        else:
            symb_counter[kqcombo] += 1
    simpler_reps = {}
    for k, v in symb_counter.items():
        if v == 1:
            only = sp.Symbol('B_{%d,%d}' %k)
            real = sp.Symbol('B_{%d,%d}^r' %k)
            imag = sp.Symbol('B_{%d,%d}^i' %k)
            simpler_reps[real] = only
            simpler_reps[imag] = only
    return Bexpr.subs(simpler_reps, simultaneous=True)

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

def Yrot(l,m,theta,phi):
    return RYlm(l, m, alpha, beta, gamma, detRot)

regen_fname = os.path.join(module_dir,'data','CPGs.pkl')
crystal_fields_fname = os.path.join(module_dir,'data','CPGs.pkl')

if os.path.exists(regen_fname):
    print("Reloading %s ..." % regen_fname)
    CPGs = pickle.load(open(os.path.join(module_dir,'data','CPGs.pkl'),'rb'))
else:
    print("%s not found, regenerating ..." % regen_fname)
    CPGs = CPGroups(group_data)
    pickle.dump(CPGs, open(os.path.join(module_dir,'data','CPGs.pkl'),'wb'))

################################## Others #################################
###########################################################################
