#!/usr/bin/env python

import sympy as sp
import numpy as np
import os
import pickle
import pandas as pd
from itertools import product
from functools import reduce
from collections import OrderedDict, Counter
from sympy.physics.quantum import Bra, Ket, KetBase, TensorProduct
from matplotlib import pyplot as plt
from qdef.constants import *
from qdef.misc import *
from IPython.display import Math
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
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
ionic_radii_df = pd.read_csv(os.path.join(module_dir,'data','ionic_radii.csv'),comment='#')

atom_symbs   = list(symb_to_name.keys())
atom_names   = list(name_to_num.keys())
GT_CGs = pickle.load(open(os.path.join(module_dir,'data',
                                            'GT_CG.pkl'),'rb'))
CG_coeffs_partitioned = pickle.load(open(os.path.join(module_dir,'data',
                                            'CG_coeffs_partitioned.pkl'),'rb'))

nistdf_levels = pd.read_pickle(os.path.join(module_dir,'data',
                                    'nist_atomic_spectra_database_levels.pkl'))
nistdf_lines = pd.read_pickle(os.path.join(module_dir,'data',
                                    'nist_atomic_spectra_database_lines.pkl'))

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

class odict(OrderedDict):
    pass

# =============================================================== #
# ===================== Load group theory data ================== #

group_dict = pickle.load(open(os.path.join(module_dir,'data','gtpackdata.pkl'),'rb'))
double_group_dict = pickle.load(open(os.path.join(module_dir,'data','gtpackdata_double_groups.pkl'),'rb'))
vcoeffs = pickle.load(open(os.path.join(module_dir,'data','Vcoeffs.pkl'),'rb'))
group_data = group_dict['group_data']
double_group_data = double_group_dict['group_data']
metadata = group_dict['metadata']
double_group_metadata = double_group_dict['metadata']
# symmetry_bases = pickle.load(open(os.path.join(module_dir,'data',
#                 'symmetry_bases.pkl'),'rb'))['symmetry_bases']

# ===================== Load group theory data ================== #
# =============================================================== #

class DetKet(KetBase):
    lbracket_latex = r'\left|'
    rbracket_latex = r'\right|\rangle'

class DetBra(KetBase):
    lbracket_latex = r'\langle\left|'
    rbracket_latex = r'\right|'

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

    def vec_in_detbasis(self, basis):
        '''
        Given an ordered basis  return  a  list  with  the
        coefficients of the qet in that basis.
        In comparison with vec_in_basis this method allows
        for  the  possibility  that the quantum numbers of
        the qet are found in some permutation in the given
        basis (adding the adequate sign).
        
        Parameters
        ----------
        basis  (list): containing the quantum numbers that
        determine  a  determinantal  basis.  These quantum
        numbers are enclosed in tuples.

        Returns
        -------
        coeffs (list): a list containing the coefficients
        for the qet in the given determinantal basis.
        '''
        coeffs = [0]*len(basis)
        for qnums, coeff in self.dict.items():
            if len(qnums) != len(set(qnums)):
                continue
            try:
                target_idx = basis.index(qnums)
                coeffs[target_idx] += coeff
            except:
                target_idx, multiplier = inspector_gadget(qnums, basis)
                coeffs[target_idx] += coeff*multiplier
        return coeffs

    def valsubs(self, subs_dict, simultaneous=True):
        '''
        The substitutions in subs_dict are evaluated on the
        coeffients of the qet.
        '''
        new_dict = dict()
        for key, val in self.dict.items():
            new_dict[key] = sp.S(val).subs(subs_dict, simultaneous=simultaneous)
        return Qet(new_dict)

    def tensor_basis_change(self, basis_changer):
        new_qet = Qet({})
        for qnums, coeff in self.dict.items():
            extra = [(Qet({(γ,):1}) if γ not in basis_changer else basis_changer[γ]) for γ in qnums]
            extra = reduce(lambda x,y: x*y, extra)
            new_qet = new_qet + ( coeff * extra)
        return new_qet

    def keysubs(self, subs_dict, simultaneous=True):
        '''
        The substitutions in subs_dict are evaluated on the
        keys of the qet.
        It assumes that they keys are already symby symbols or
        tuple of them.
        '''
        new_dict = dict()
        for key, val in self.dict.items():
            if isinstance(key, tuple):
                new_key = tuple((k.subs(subs_dict, simultaneous=simultaneous) for k in key))
            else:
                new_key = key.subs(subs_dict, simultaneous=simultaneous)
            if new_key in new_dict:
                new_dict[new_key] += sp.S(val)
            else:
                new_dict[new_key] = sp.S(val)
        return Qet(new_dict)

    def __mul__(self, multiplier):
        '''
        A  qet can be multiplied by a scalar or by another
        qet.  When  two  qets  are  multiplied the implied
        multiplication   is  taken  as  a  tensor  product
        between the two qets.
        '''
        if isinstance(multiplier, Qet):
            new_dict = dict()
            for k1, v1 in self.dict.items():
                for k2, v2 in multiplier.dict.items():
                    k3 = k1 + k2
                    v3 = v1 * v2
                    if v3 != 0:
                        new_dict[k3] = v3
            return Qet(new_dict)
        else:
            new_dict = dict(self.dict)
            for key, coeff in new_dict.items():
                new_dict[key] = multiplier*(coeff)
            return Qet(new_dict)

    def __rmul__(self, multiplier):
        '''
        This is required to enable multiplication
        from the left and from the right.
        '''
        new_dict = dict()
        for key, coeff in self.dict.items():
            new_dict[key] = multiplier*(coeff)
        return Qet(new_dict)

    def simplify(self):
        '''simplify coefficients'''
        new_dict = dict(self.dict)
        for key, coeff in new_dict.items():
            new_dict[key] = sp.simplify(coeff)
        return Qet(new_dict)

    def parasimplify(self):
        '''simplify coefficients using several cores'''
        new_dict = dict(self.dict)
        tuple_simplify = lambda key,coeff: (key, sp.simplify(coeff))
        simp_dict = Parallel(n_jobs = num_cores)(delayed(tuple_simplify)(key, coeff) for key, coeff in new_dict.items())
        return Qet(dict(simp_dict))
    
    def basis(self):
        '''return a list with all the keys in the qet'''
        return list(self.dict.keys())

    def dual(self):
        '''conjugate all the coefficients'''
        new_dict = dict(self.dict)
        for key, coeff in new_dict.items():
            new_dict[key] = sp.conjugate(coeff)
        return Qet(new_dict)

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

    def as_bra(self, fold_keys=False, nice_negatives=False):
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
                    sympyRep += coeff*Bra(*key)
                else:
                    sympyRep += coeff*Bra(key)
        return sympyRep

    def as_operator(self, opfun):
        OP = sp.S(0)
        for key, val in self.dict.items():
                if isinstance(key,tuple):
                    OP += sp.S(val) * opfun(*key)
                else:
                    OP += sp.S(val) * opfun(key)
        return OP

    def as_detbraket(self):
        '''
        Give  a  representation  of  the qet as a detBra*detKet. The
        keys in the dict for the ket are assumed to split first half
        for the detbra, and other second half for the detket.
        '''
        sympyRep = sp.S(0)
        for key, coeff in self.dict.items():
            l = int(len(key)/2)
            if key == ():
                sympyRep += coeff
            else:
                sympyRep += coeff*(DetBra(*key[:l])*DetKet(*key[l:]))
        return sympyRep
    
    def as_spinorb_ket(self):
        '''
        Give   a   representation   of   the   qet  as  a  Ket  from
        sympy.physics.quantum  assuming  that  its  quantum  numbers
        represent spin orbitals for a spin 1/2 particle.
        '''
        sympyRep = sp.S(0)
        for key, coeff in self.dict.items():
            if key == ():
                sympyRep += coeff
            else:
                if isinstance(key, tuple):
                    symbs = [skey.orbital if skey.spin == S_UP else sp.Symbol(r'\bar{%s}' % sp.latex(skey.orbital)) for skey in key]
                    sympyRep += coeff * Ket(*symbs)
                else:
                    if key.spin == S_UP:
                        sympyRep += coeff * Ket(key.orbital)
                    else:
                        sympyRep += coeff * Ket(sp.Symbol(r'\bar{%s}' % sp.latex(key.orbital)))
        return sympyRep


    def as_detbra(self):
        '''
        Give a representation of the qet  as  a  determinantal
        Bra  from sympy.physics.quantum.
        This is for presentation only, don't use for calculations.
        '''
        sympyRep = sp.S(0)
        for key, coeff in self.dict.items():
            if key == ():
                sympyRep += coeff
            else:
                sympyRep += coeff*DetBra(*key)
        return sympyRep
    
    def as_detket(self):
        '''
        Give a representation of the qet  as  a  determinantal
        Ket  from sympy.physics.quantum.
        This is for presentation only, don't use for calculations.
        '''
        sympyRep = sp.S(0)
        for key, coeff in self.dict.items():
            if key == ():
                sympyRep += coeff
            else:
                sympyRep += coeff*DetKet(*key)
        return sympyRep

    def as_braket(self):
        '''
        Give a representation of the qet as a Bra*Ket. The
        keys in the dict for the ket are assumed to  split
        first half for the bra, and other second half  for
        the ket.
        This is for presentation only, don't use for calculations.
        '''
        sympyRep = sp.S(0)
        for key, coeff in self.dict.items():
            l = int(len(key)/2)
            if key == ():
                sympyRep += coeff
            else:
                sympyRep += coeff*(Bra(*key[:l])*Ket(*key[l:]))
        return sympyRep

    def apply(self, f):
        '''
        This method can be used to apply a function  to   a
        qet. The provided function f must take as arguments
        a single pair of  (qnums, coeff)  and   return    a
        dictionary or a (qnum, coeff) tuple.
        '''
        new_dict = dict()
        for key, coeff in self.dict.items():
            appfun = f(key, coeff)
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

    def norm(self):
        '''
        Compute the norm of the qet.
        '''
        norm2 = 0
        for key, coeff in self.dict.items():
            norm2 += abs(coeff)**2
        return sp.sqrt(norm2)

    def normalized(self):
        '''
        Return a normalized version of qet.
        '''
        thenormalizer = sp.S(1)/self.norm()
        return thenormalizer*Qet(self.dict)
    
    def permute(self, permutation):
        '''
        Keys of qet must be tuples for this to work,
        given permutation is a list of indices to which
        the initial orderding is to be mapped to.
        '''
        new_dict = {}
        for k,v in self.dict.items():
            new_k = tuple([k[p] for p in permutation])
            new_dict[new_k] = v
        return Qet(new_dict)
 
    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return 'Qet(%s)' % str(self.dict)

def opSingleMulti(Opdict, pos):
    '''
    Returns a function that can be used to evaluate the result of applying
    a single electron operator on a multi-electron qet.
    The operator acting only on the orbital part of the wavefunction.

    Parameters
    ----------
    pos  (int): the slot of the wave function where the single electron is
    located.  
    Opdict (dict): keys equal to ml values that label basis orbital states
    and  values equal to dictionaries. These dictionaries having ml values
    as  keys  and values equal to the corresponding coefficients of having
    applied the related operator.

    Returns
    -------
    multifun (func): a function that can be applied to a qet whose keys are
    of the form ((ml1, ms1), (ml2, ms2), ...).
    '''
    def multifun(qnums, coeff):
        mlmsL, mlmsi, mlmsR = list_splitter(qnums, pos)
        (mli, msi) = mlmsi
        qet_dict = Opdict[mli]
        qdict = {}
        for key, val in qet_dict.items():
            skey = mlmsL + ((key,msi),) + mlmsR
            if skey in qdict:
                qdict[skey] += val*coeff
            else:
                qdict[skey] = val*coeff
        return qdict
    return multifun

class HartreeFockData():
    '''
    Free-ion   data   from   Fraga's   "Handbook   of  Atomic  Data"
    complemented  with  approximations  for dealing with ions placed
    inside of crystals as obtained from Morrison's "Angular Momentum
    Theory Applied to Interactions in Solids".

    The  data  computed  by  Fraga  are the results of calculcations
    using  non-relativistic functions using a numerical Hartree-Fock
    program (Froese-Fisher (1969)).
    '''
    HFradavg = pickle.load(open(os.path.join(module_dir,'data','HF_radial_avgs.pkl'),'rb'))
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
    symb_to_num = symb_to_num

    @classmethod
    def radial_average(cls, element, charge_state, n, nephelauxetic=False):
        '''
        Returns  the radial average <r^n> for a valence electron for
        the  given element and charge state (n=0 neutral, n=1 singly
        ionized, ...) within the limitations of Hartree-Fock.

        The  element  can be given either as its atomic number or by
        its symbol.

        Data is taken from Fraga's et al Handbook of Atomic Data.

        The unit for the provided radial average is Angstrom^n.

        Provided data has 5 significant figures.

        Parameters
        ----------
        element (int or str): atomic number of element symbol.
        charge_state   (int): how many electrons the ion is missing.
        n              (int): as in <r^n>.

        nephelauxetic  (bool):  whether  the nephelauxetic effect is
        considered.

        Returns
        -------
        rad_avg  (float): if nephelauxetic is True value returned is
        <r^n>_HF  /  τ^n,  else  what  is  returned  is  <r^n>_HF as
        obtained  from  Fraga's  "Handbook  of  Atomic  Data"  using
        Hartree-Fock methods.

        '''
        charge_state = int(charge_state)
        num_charge_state = charge_state
        assert charge_state >= 0, "What odd ion state you speak of?"
        charge_state = cls.ArabicToRoman[charge_state+1]
        if isinstance(element, int):
            element = cls.num_to_symb[element]
        try:
            rad_avg = float(cls.HFradavg['<r^%d>' % n].loc[[element]][charge_state])
            tau = 1
            if nephelauxetic:
                tau = cls.nephelauxetic_factor_tau(element, num_charge_state)
            rad_avg = rad_avg / tau**n
            return rad_avg
        except:
            raise ValueError('This radial average is not here.')
    
    @classmethod
    def atom_size(cls, element, charge_state, nephelauxetic=False):
        '''
        Parameters
        ----------
        element (str or int): atomic number of symbol for an element
        charge_state (int)  : charge state of atom
        nephelauxetic  (bool): if True then the returned size is the
        estimated  one  for  the  ion inside of a crystal. This is a
        rough approximation.

        Returns
        -------
        radius (float): estimated radius in Angstrom
        
        References
        ----------
        + Fraga, Karwowski, and Saxena, “Handbook of Atomic Data.”
        + Morrison, "Angular Momentum Theory Applied to Interactions
        in Solids".
        '''
        if isinstance(element, int):
            element = cls.num_to_symb[element]
        charge_state = cls.ArabicToRoman[charge_state+1]
        τ = 1.
        if nephelauxetic:
            τ = cls.nephelauxetic_factor_tau(element, charge_state)
        radius = float(cls.HFsizes.loc[[element]][charge_state]) / τ
        return radius
    
    @classmethod
    def nephelauxetic_factor_tau(cls, element, charge_state):
        '''
        One  approximation  to  atomic  wavefunctions  for electrons
        placed  inside of crystals is simply to scale the hydrogenic
        wavefunctions by a scaling factor tau. This function returns
        that factor as presented in Table 13.4 of Morrison (1988).

        Note  that  there  is  one  typo  in  Morrison that has been
        corrected here. (0.11128 in there should be 0.011128)

        Parameters
        ----------
        element (int)    : Atomic number.
        charge_state(int):

        Returns
        -------
        tau (float)

        References
        ----------
        + Morrison,  Angular Momentum Theory Applied to Interactions
        in Solids.
        '''
        if not isinstance(element, int):
            element = cls.symb_to_num[element]
        if charge_state == 2:
            N = element - 20
            tau =  0.76878  + 0.011128 * N
        elif charge_state == 3:
            N = element  - 21
            tau = 0.8118355958 + 0.007398583637 * N
        else:
            N = element  - 22
            tau = 0.833540 + 0.0056609 * N
        assert N in range(1,10), "Invalid configuration."
        return tau

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
        self.nist_data_levels = nistdf_levels[nistdf_levels['Element'] == self.symbol]
        # a dataframe with line data as compiled by NIST
        self.nist_data_lines = nistdf_lines[nistdf_lines['element'] == self.symbol]
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
        datum = self.nist_data_levels[self.nist_data_levels['Charge'] == charge]
        energy_levels = datum['Level (eV)']
        configs = datum['Configuration']
        if charge == 0:
            fig_name = '%s' % (self.symbol)
            latex_name = fig_name
        else:
            fig_name = '%s +%d' % (self.symbol, charge)
            latex_name = '\mathrm{%s}^{+\!%d}' % (self.symbol, charge)
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
        ax.set_ylabel('E / eV')
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
        self.nist_data_levels = self.nist_data_levels[self.nist_data_levels['Charge'] == self.charge_state]
        self.nist_data_levels.reset_index(drop=True, inplace=True)
        self.nist_data_lines = self.nist_data_lines[self.nist_data_lines['charge'] == self.charge_state]
        self.nist_data_lines.reset_index(drop=True, inplace=True)
    
    def ionic_radius(self, radius_type='crystal'):
        '''
        Parameters
        radius_type (str): either 'crystal' or 'effective'
        Returns
        If found in data returned value is the radius in pm, if not the function returns None.
        '''
        results = ionic_radii_df[(ionic_radii_df.loc[:,'symbol'] == self.symbol) & 
                        (ionic_radii_df.loc[:,'radius type'] == radius_type) & 
                        (ionic_radii_df.loc[:,'charge'] == self.charge_state)]
        if len(results) > 0:
            return float(list(results['radius/pm'])[0])
        else:
            return None

    def level_diagram(self, min_energy=-np.inf, max_energy=np.inf):
        charge = self.charge_state
        '''make a nice plot of the levels of the ion with the given charge'''
        cmap = plt.cm.RdYlGn
        datum = self.nist_data_levels[self.nist_data_levels['Charge'] == charge]
        energy_levels = datum['Level (eV)']
        configs = datum['Configuration']
        if charge == 0:
            fig_name = '%s' % (self.symbol)
            latex_name = fig_name
        else:
            fig_name = '%s +%d' % (self.symbol, charge)
            latex_name = '\mathrm{%s}^{+\!%d}' % (self.symbol, charge)
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
        ax.set_ylabel('E / eV')
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
        '''
        Parameters
        ----------
        fig  (plt.fig):  passed  by  reference  to  be caught as the
        return of this method
        annotations (dict): keyed either by atomic number or symbol,
        values are strings that are shown below the element symbols.
        Returns
        -------
        fig, ax
        '''
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

class SpinOrbital():
    '''
    Class used to represent a spin orbital.
    '''
    def __init__(self, orbital_part, spin):
        '''
        Parameters
        ----------
        orbital_part (anything, usually sp.Symbol)
        spin   (S_DOWN or S_UP)

        '''
        self.orbital = orbital_part
        self.spin = spin

    def __repr__(self):
        if self.spin == S_DOWN:
            return str(self.orbital) + '↓'
        else:
            return str(self.orbital) + '↑'
    
    def _repr_latex_(self):
        if isinstance(self.spin, tuple):
            bits = ['\\bar{%s}' % orb if (spinner == S_DOWN) else str(orb) for orb, spinner in zip(self.orbital, self.spin)]
            return '$\\displaystyle (%s)$' % ','.join(bits)
        else:
            if self.spin == S_UP:
                return '$\\displaystyle %s$' % str(self.orbital)
            else:
                return '$\\displaystyle \\bar{%s}$' % str(self.orbital)
    
    def __eq__(self, other):
        return (self.spin == other.spin and self.orbital == other.orbital)
    
    def __hash__(self):
        return hash((self.orbital, self.spin))

class Term():
    '''
    A term containing states that correspond to it.
    A few methods are given to create readable representations.

    Term  is  initialized  with  a  dictionary  whose  keys  are
    namedtuples  labeling  the  term states and values which are
    qets that represent the states in a certain basis.

    Attributes
    ----------
    term_prototype (str): use to create symbol representing the term.
    state_label_prototype (str): use to create symbols representing states.
    '''
    def __init__(self, init_dict):
        for k,v in init_dict.items():
            setattr(self, k, v)
        self.term_prototype = sp.Symbol(r'{{}}^{{{M}}}{ir}')
        self.state_label_prototype = sp.Symbol(r'\Psi({α}, {S}, {{{Γ}}}, {M_s}, {γ})')
    
    def term_symbol(self):
        '''
        Create sympy Symbol that represents the term symbol.
        '''
        return sp.Symbol(str(self.term_prototype).format(M=str(2*self.S+1),ir=str(sp.latex(self.irrep))))
    
    def make_state_symbols(self):
        '''
        Create LaTeX strings for the all the states in the term.
        Returns
        -------
        state_symbols: (list) [sp.Symbol, ...]
        '''
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

def orthogonal_matrix(euler_params):
    '''
    For a given symmetry operation as parametrized by Euler angles
    α, β, γ, and by its  determinant  det (±1).  The corresponding
    orthogonal matrix is returned.

    The  notation  used  here matches the notation used in Bradley
    and Cracknell page 52.

    This corresponds to the  following sequence of rotations about
    fixed axes z-y-z:

    - A first rotation about the z-axis by α,
    - a second rotation about the y-axis by β,
    - and final rotation about the z-axis by γ.

    R_z(γ) * R_y(β) * R_z(α)

    '''
    α, β, γ, det = euler_params

    row_0 = [-sp.sin(α)*sp.sin(γ) + sp.cos(α)*sp.cos(β)*sp.cos(γ),
             -sp.cos(α)*sp.sin(γ) - sp.cos(γ)*sp.sin(α)*sp.cos(β),
              sp.sin(β)*sp.cos(γ)]
    row_1 = [sp.cos(α)*sp.cos(β)*sp.sin(γ) + sp.cos(γ)*sp.sin(α),
            -sp.sin(α)*sp.sin(γ)*sp.cos(β) + sp.cos(α)*sp.cos(γ),
            sp.sin(γ)*sp.sin(β)]
    row_2 = [-sp.cos(α)*sp.sin(β),
             sp.sin(α)*sp.sin(β),
             sp.cos(β)]
    mat = det*sp.Matrix([row_0,row_1,row_2])
    return mat

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
    def __repr__(self):
        return self.odict.__repr__()

class CrystalGroup():
    '''
    Class  for  a  point  symmetry  group  which can be initialized with a
    string-label or corresponding standard index.

    The class has the following attributes:

        - index (int): index for the group in the standard order.

        - label (str): string-label for the group

        -  classes  (dict):  keys being equal to the different classes tha
        the  group  has  and  values  being  equal  to lists that have the
        symbols  representing  the  symmetry  elements  that belong to the
        classes.

        -  class_labels  (list):  containing  symbols  for  the  different
        classes of the group.

        -  group_operations  (list):  containing  symbols  for  the  group
        operations.

        -  irrep_labels  (list):  containing  symbols  for the irreducible
        representations of the group.

        - character_table (sp.Matrix): character table for the group where
        the  columns  of the given matrix correspond to the classes of the
        group  (in  the  order given by the .classes_labels attribute) and
        the  rows  correspond  to  the  irreducible representations of the
        group (in the order given by the .irrep_labels attribute).

        -  character_table_inverse  (sp.Matrix):  matrix  inverse  of  the
        character  table  of  the  group. Useful for decomposing reducible
        representations into its irreducible components.

        -  irrep_matrices  (dict): keys equal to symbols for the different
        irreducible representations of the group and values being equal to
        dictionaries  whose  keys are symbols for the different operations
        of  the  group  and  values  being  equal  to matrices that form a
        corresponding irreducbible matrix representation of the group.

        -  generators  (list):  containing the symbols of group operations
        that can be used as generators for the group.

        -  multiplication_table  (sp.Matrix):  each element represents the
        result  of  multiplying  two of the elements a⋅b of the group, the
        different   rows   representing   a   and  the  different  columns
        representing b.

        -   euler_angles   (dict):   with  keys  being  equal  to  symbols
        representing  group operations and values equal to lists with four
        elements  [α,  β,  γ, det] with α the angle for the first rotation
        about  the  z-axis,  β the angle for the second rotation about the
        y-axis,  γ  the  angle of the final rotation about the z-axis, and
        det  being  equal  to  ±1 the determinant of the orthogonal matrix
        representing   the   group   operation.  Angles  correspond  to  a
        convention of an Euler angle rotation with fixed z-y-z axes.

        -  order  (int):  order  of  the group, which is how many symmetry
        operations the group has.

        -  product_table  (qd.ProductTable):  contains  the  reductions to
        which    every   binary   product   of   the   group   irreducible
        representations resolve to.

        -  operation_matrices  (dict):  with keys equal to symbols for the
        group operations and values being equal to the orthogonal matrices
        that   represent  the  operation  as  geometrical  transformations
        carried out in three dimensions.

        -  irrep_dims  (dict): keys being equal to symbols for irreducible
        representations  and values being integers equal to the dimensions
        of the corresponding irrep.

        -  CG_coefficients  (dict):  keys  are equal to triples (c_0, c_1,
        c_2)   with  symbols  for  three  components  of  the  irreducible
        components   of  the  group  and  values  equal  to  the  coupling
        coefficient for c_2 from having coupled c_0 and c_1. If the key is
        absent, that means that the coefficient is zero.

        -  CG_coefficients_partitioned (dict): a dictionary whose keys are
        tuples of the form (ir0, ir1) with ir0 and ir1 two symbols for two
        irreducible  representation  of  the  group and the values being a
        dictionary   that  has  all  the  coupling  coefficients  for  the
        components  of  ir0 and ir1 into the components of the irreps into
        which ir0Xir1 decomposes to.
    '''
    def __init__(self, group_data_dict, double_group_data_dict: None):
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
        self.operation_matrices = {k: orthogonal_matrix(v) for k, v in self.euler_angles.items()}
        self.irrep_dims = {k: list(v.values())[0].shape[0] for k, v in self.irrep_matrices.items()}
        self.product_table = self.direct_product_table(self.irrep_labels, self.character_table, self.character_table_inverse, self.label )
        self.component_labels = self.get_component_labels()
        # self.symmetry_adapted_bases = symmetry_bases[self.label]
        self.CG_coefficients = GT_CGs[self.label]
        self.CG_coefficients_partitioned = CG_coeffs_partitioned[self.label]
        self.gen_char_table_dict()
        if self.label in vcoeffs.keys():
            self.V_coefficients = vcoeffs[self.label]
        else:
            self.V_coefficients = None
        if double_group_dict:
            self.double_classes = double_group_data_dict['classes']
            self.double_irrep_labels = double_group_data_dict['irrep labels']
            self.double_character_table = double_group_data_dict['character table']
            self.double_character_table_inverse = double_group_matrix_inverse(double_group_data_dict['character table'], self.label)
            self.double_class_labels = double_group_data_dict['class labels']
            self.double_multiplication_table = double_group_data_dict['multiplication table']
            self.double_euler_angles = double_group_data_dict['euler angles']
            self.double_group_operations = double_group_data_dict['group operations']
            self.double_order = len(self.double_group_operations)
            self.double_operation_matrices = {k: orthogonal_matrix(v) for k, v in self.double_euler_angles.items()}
            self.double_irrep_dims = {k : self.double_character_table[self.double_irrep_labels.index(k),0] for k in self.double_irrep_labels}
            self.double_product_table = self.direct_product_table(self.double_irrep_labels, self.double_character_table, self.double_character_table_inverse, self.label )
            # self.double_component_labels = self.get_component_labels()
            # self.double_symmetry_adapted_bases = double_symmetry_bases[self.label]

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

    def direct_product(self, ir0, ir1, group_irreps, group_chartable, char_table_inverse):
        '''
        Given the label for a cpg and  labels for  two
        of its irreducible  representations, determine
        the direct sum decomposition of their product.
        This product  is  returned  as a qet with keys
        corresponding  to  the irreps and values equal
        to the integer coefficients.
        '''
        # grab group classes, irrep names, and character table
        assert ir0 in group_irreps, 'irrep not in %s' % str(group_irreps)
        assert ir1 in group_irreps, 'irrep not in %s' % str(group_irreps)
        chars_0, chars_1 = [group_chartable.row(group_irreps.index(ir)) for ir in [ir0, ir1]]
        chars = sp.Matrix([char0*char1 for char0, char1 in zip(chars_0, chars_1)])
        partition = (char_table_inverse*chars)
        qet = Qet()
        for element, ir in zip(partition, group_irreps):
            el = int(sp.N(element,1,chop=True))
            qet = qet + Qet({ir:el})
        return list(Counter((qet.dict)).elements())

    def direct_product_table(self, group_irreps, char_table, char_table_inverse, group_label):
        '''
        This creates the complete set of  binary
        products of irreducible  representations
        for the given group.
        The result is saved as  an attribute  in
        the group.
        '''
        product_table = OrderedDict()
        for ir0 in group_irreps:
            for ir1 in group_irreps:
                if (ir1,ir0) in product_table.keys():
                    product_table[(ir0,ir1)] = product_table[(ir1,ir0)]
                else:
                    product_table[(ir0,ir1)] = self.direct_product(ir0, ir1, group_irreps, char_table, char_table_inverse )#.as_symbol_sum()
        return ProductTable(product_table, group_irreps, group_label)

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label

class CPGroups():
    '''
    Class  that  aggregates all  the  crystallographic point groups into a
    single object having the following attributes:

    all_group_labels (list): a list having all the string labels  for  the 
    32 crystallographic point groups.
    groups   (dict):   with   keys   equal   to   string  labels  for  the
    crystallographic    point   groups   and   values   being   equal   to
    qd.CrystalGroup.
    '''
    def __init__(self, groups, doublegroups):
        self.all_group_labels = [
            'C_{1}',  'C_{i}',  'C_{2}',  'C_{s}',
            'C_{2h}', 'D_{2}',  'C_{2v}', 'D_{2h}',
            'C_{4}',  'S_{4}',  'C_{4h}', 'D_{4}',
            'C_{4v}', 'D_{2d}', 'D_{4h}', 'C_{3}',
            'S_{6}',  'D_{3}',  'C_{3v}', 'D_{3d}',
            'C_{6}',  'C_{3h}', 'C_{6h}', 'D_{6}',
            'C_{6v}', 'D_{3h}', 'D_{6h}', 'T',
            'T_{h}',  'O',      'T_{d}',  'O_{h}']
        self.groups = {}
        for k in groups:
            print('Parsing %s ...' % self.all_group_labels[k-1])
            self.groups[k] = CrystalGroup(groups[k], doublegroups[k])
        self.metadata = metadata
    def get_group_by_label(self, label):
        if '{' not in label:
            if len(label) > 1:
                label = '%s_{%s}' % label.split('_')
        group_idx = 1 + self.all_group_labels.index(label)
        return self.groups[group_idx]