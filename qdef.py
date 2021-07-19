import os, re, pickle
import numpy as np
from math import ceil

import sympy as sp
from sympy import pi, I
from sympy.physics.quantum import Ket, Bra

from collections import OrderedDict
from itertools import product
from tqdm.notebook import tqdm

from IPython.display import display, HTML, Math

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
        new_dict = dict()
        for key, val in self.dict.items():
            new_dict[key] = sp.S(val).subs(subs_dict)
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
        tot = sp.sp.S(0)
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
        '''at times a tuple of dict needs to
        be identified with its inverse'''
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

    def __repr__(self):
        return str(self.dict)

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

# =========================== Classes =========================== #
# =============================================================== #

###########################################################################
#################### Calculation of Surface Harmonics #####################

def SubSupSymbol(radix,ll,mm):
    '''
    Generates a symbol placeholder for the B coefficients in the crystal field potential.
    '''
    SubSupSym = sp.symbols(r'{%s}_{%s}^{%s}' % (radix, str(ll), str(mm)))
    return SubSupSym

def SubSubSymbol(radix,ll,mm):
    '''
    Generates a symbol placeholder for the B coefficients in the crystal field potential.
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
      Wig_D_1 = (sp.cos(-n*gamma-m*alpha)+I*sp.sin(-n*gamma-m*alpha)) * Wigner_d(l,n,m,beta)
      Wig_D = Wig_D_0 * Wig_D_1
      Wig_D = Wig_D
    return Wig_D
Wigner_D.values = {}

def real_or_imagined(qet):
    '''
    For a given superposition of spherical  harmonics,  determine  if  the
    total has a pure imaginary (i), pure real (r), or mixed character (m),
    it assumes that the coefficients in the superposition are all real.
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
    This would be rotateHarmonic in the Mathematica  code.  It  is used
    in the projection of the spherical  harmonics  to  create  symmetry
    adapted wavefunctions.
    '''
    Rf = Qet()
    for nn in range(-l,l+1):
        wigD = Wigner_D(l, m, nn, alpha, beta, gamma)
        if wigD != 0:
          Rf = Rf + Qet({(l,nn): wigD})
    return (sp.S(detRot)**l) * Rf

def flatten_matrix(mah):
    '''
    A convenience function
    to flatten a sympy matrix into a
    list of lists
    '''
    return [item for sublist in mah.tolist() for item in sublist]

def SymmetryAdaptedWF(group, l, m):
    '''
    This returns the proyection of Y_l^m
    on the trivial irreducible representation
    of the given group
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
        SALC += RYlm(l,m,alpha,beta,gamma,detRot)
    SALC = (sp.S(1)/order)*SALC
    SALC = SALC.apply(lambda x,y : (x, sp.simplify(y)))
    return SALC

def linearly_independent(vecs):
    '''
    Given a list of vectors
    return the largest subset which
    of linearly independent ones
    and the indices that correspond
    to them in the original list.
    '''
    matrix = sp.Matrix(vecs).T
    good_ones = matrix.rref()[-1]
    return good_ones, [vecs[idx] for idx in good_ones]

def SymmetryAdaptedWFs(group, l, normalize=True, verbose=False, sympathize=True):
    '''
    For a given group and a given value of
    l, this returns a set of linearly independent
    symmetry adapted functions which are also real-valued.
    If the set that is found initially contains combinations that are
    not purely imaginary or pure real, then the assumption
    is made that this set contains single spherical
    harmonics, and then sums and differences between
    m and -m are given by doing this through the values
    of |m| for the functions with mixed character.
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

#################### Calculation of Surface Harmonics #####################
###########################################################################

###########################################################################
############### Calculation of Clebsch-Gordan Coefficients ################

def cg_symbol(comp_1, comp_2, irep_3, comp_3):
    '''
    Given symbols for three components (comp_1, comp_2, comp_3) of three
    irreducible representations of a  group   this  function  returns  a
    sp.Symbol for the corresponding Clebsch-Gordan coefficient:

    <comp_1,comp_2|irep_3, comp_3>

    The symbol of the third irrep is given  explicitly  as  irep_3.  The
    other two irreps are implicit in (comp_1) and (comp_2) and should be
    one of the symbols in group.irrep_labels.

    (comp_1, comp_2, and comp_3) may be taken as elements from
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
    Given a group and symbol labels for two irreducible representations
    Γ1 and Γ2 this function calculates the  Clebsh-Gordan  coefficients
    used to span the basis functions of  their  product in terms of the
    basis functions of their factors.

    By  assuming  an even phase convention for all coefficients the
    result is also given for the exchanged order (Γ2, Γ1).

    If rep_rules = False, this  function  returns  a tuple with 3  ele-
    ments, the first element being a matrix of symbols for the CGs coe-
    fficients for (Γ1, Γ2)  the  second element  the  matrix  for  sym-
    bols  for (Γ2, Γ1) and the third one being a matrix  to  which  its
    elements are matched element by element to  the  first  and  second
    matrices of symbols.

    If rep_rules = True, this  function  returns     two  dictionaries.
    The keys in the first one equal CGs coefficients from (Γ1, Γ2)  and
    the second one those for (Γ2, Γ1); with the values  being  the  co-
    rresponding coefficients.

    These CG symbols are made according to the following template:
        <i1,i2|i3,i4>
        (i1 -> symbol for basis function in Γ1 or Γ2)
        (i2 -> symbol for basis function in Γ2 or Γ1)
        (i3 -> symbol for an irreducible representation Γ3 in the group)
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

def Ckq2THI(l,m,theta,phi):
    return ThreeHarmonicIntegrate(l1, m1, l, m, l3, m3)

def ThreeHarmonicIntegrate(l1, m1, l2, m2, l3, m3):
    '''
    ThreeHarmonicIntegrate(l1,m1,l2,m2,l3,m3) solves the integral of
    the product of three spherical harmonics (i.e., eqn 1.15 from  STK) where
    one is unnormalized. This is used in the determination of the single
    electron crystal field splitting.

    l1 : l quantum number of orbital 1 (Y_l1,m1) [l in STK notation]
    l3 : l quantum number of orbital 2 (Y_l3,m3) [l' in STK notation]
    m1 : m quantum number of l1 [-l1,-l1+1,...,l1-1,l1] [m in STK notation]
    m3 : m quantum number of l3 [-l3,-l3+1,...,l3-1,l3] [m' in STK notation]

    l2 : k in STK notation [ |l-l'|<=k<=l+l' ]
    m2 : q in STK notation [ q = m-m' ]
    '''
    from sympy.physics.quantum.cg import CG
    THI = sp.sqrt((2*l1+1)/(2*l3+1))*CG(l2,0,l3,0,l1,0)*CG(l2,m2,l3,m3,l1,m1)
    return THI

def SingleElectronSplitting(Group, l=4, orbital = 'd', debug=False):
    '''
    ----------------------------
    | orbitals | s | p | d | f |
    ----------------------------
    |    l     | 0 | 1 | 2 | 3 |
    ----------------------------
    Note that this is currently only valid for intra-orbital
    '''

    if isinstance(orbital,str):
        if orbital == 's':
            orbital = 0
        elif orbital == 'p':
            orbital = 1
        elif orbital == 'd':
            orbital = 2
        elif orbital == 'f':
            orbital = 3
        else:
            print('ERROR: Orbital does not exist! Please choose one of s,p,d, or f orbitals')

    global l1, m1, l3, m3

    l1 = sp.Symbol('l1')
    m1 = sp.Symbol('m1')
    l3 = sp.Symbol('l3')
    m3 = sp.Symbol('m3')

    CFP_Table = []
    THI_func = CFP(Group,l=4).replace(Ynm,Ckq2THI)

    for mm1 in np.arange(-orbital,orbital+0.1):
        mm1 = int(mm1)
        col_tmp = []
        for mm3 in np.arange(-orbital,orbital+0.1):
            mm3 = int(mm3)
            col_tmp.append(THI_func.subs([(l1, orbital), (m1,mm1), (l3, orbital), (m3,mm3)]).simplify())
        CFP_Table.append(col_tmp)

    del l1, m1, l3, m3

    EigenSys = sp.Matrix(CFP_Table).eigenvects()
    EigenVals = []
    EigenVecs = []
    for eRes in EigenSys:
        for deg in np.arange(eRes[1]):
            EigenVals.append(eRes[0])
            EigenVecs.append(list(eRes[2][deg]))

    Yarray = []
    for mm in np.arange(-orbital,orbital+0.1):
        Yarray.append(sp.Ynm(2,int(mm),theta,phi))
    Yarray = sp.Matrix(Yarray)
    EigenVecs = sp.Matrix(EigenVecs)

    global alpha, beta, gamma, detRot

    eVec = list(EigenVecs*Yarray)
    reps = []
    ParTable = Group.ParameterTable
    for evc in eVec:
        rotEVecs = []

        for rep in np.arange(len(ParTable)):
            alpha = ParTable[rep][0]
            beta = ParTable[rep][1]
            gamma = ParTable[rep][2]
            detRot = ParTable[rep][3]
            rotEVecs.append(evc.replace(Ynm,Yrot))
        reps.append(np.array(Group.ElementCharacterTable).dot(np.array(rotEVecs)).tolist())

    repIdx = []
    for row in reps:
        rowsimp = []
        for col in row:
            colsimp = sp.simplify(col)
            CoeffDict = colsimp.as_coefficients_dict()

            col_tmp = 0
            for coeff in CoeffDict.keys():

                if abs(CoeffDict[coeff])<1e-10:
                    CoeffDict[coeff] = 0
                col_tmp = col_tmp + CoeffDict[coeff]*coeff
            rowsimp.append(col_tmp)

        repIdx.append([idx for idx, val in enumerate(rowsimp) if val != 0][0])

    '''if debug == True:
        for row in reps:
            for col in row:
                print(simplify(col))'''

    srtIdx = np.array(repIdx).argsort().tolist()
    return EigenVals, eVec, repIdx, srtIdx

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

if os.path.exists(regen_fname):
    print("Reloading %s ..." % regen_fname)
    CPGs = pickle.load(open(os.path.join(module_dir,'data','CPGs.pkl'),'rb'))
else:
    print("%s not found, regenerating ..." % regen_fname)
    CPGs = CPGroups(group_data)
    pickle.dump(CPGs, open(os.path.join(module_dir,'data','CPGs.pkl'),'wb'))
