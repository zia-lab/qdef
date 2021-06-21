import os, csv, re
import numpy as np
from collections import OrderedDict

from sympy import pi, I, Matrix, symbols, zeros, latex, simplify, N
from sympy import Symbol, linsolve, Eq, eye, solve
from sympy import S, conjugate, GramSchmidt
from sympy import Dummy, sympify, Function
from sympy.physics.quantum import Ket, Bra
from sympy import Abs, exp, sqrt, factorial, sin, cos, cot, sign
import sympy as sp

import math
import pickle

from IPython.display import display, HTML, Math

module_dir = os.path.dirname(__file__)

from sympy import Ynm

# To avoid simplification of negative m values, the eval method
# on the spherical  harmonics  Ynm  needs  to be redefined. All
# that is done is  commenting   out  a  portion of the original
# source code:

@classmethod
def new_eval(cls, n, m, theta, phi):
    n, m, theta, phi = [sympify(x) for x in (n, m, theta, phi)]
    # Handle negative index m and arguments theta, phi
    #if m.could_extract_minus_sign():
    #    m = -m
    #    return S.NegativeOne**m * exp(-2*I*m*phi) * Ynm(n, m, theta, phi)
    if theta.could_extract_minus_sign():
        theta = -theta
        return Ynm(n, m, theta, phi)
    if phi.could_extract_minus_sign():
        phi = -phi
        return exp(-2*I*m*phi) * Ynm(n, m, theta, phi)
Ynm.eval = new_eval

group_dict = pickle.load(open(os.path.join(module_dir,'data','gtpackdata.pkl'),'rb'))
group_data = group_dict['group_data']
metadata = group_dict['metadata']

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
            new_dict[key] = S(val).subs(subs_dict)
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
            new_dict[key] = conjugate(coeff)
        return Qet(new_dict)

    def as_operator(self, opfun):
        OP = S(0)
        for key, val in self.dict.items():
                OP += S(val) * opfun(*key)
        return OP

    def as_ket(self, fold_keys=False, nice_negatives=False):
        '''
        Give a representation of the qet  as  a  Ket  from
        sympy.physics.quantum, fold_keys  =  True  removes
        unnecessary parentheses and nice_negatives =  True
        assumes all numeric  keys  and  presents  negative
        values with a bar on top.
        '''
        sympyRep = S(0)
        for key, coeff in self.dict.items():
            if key == ():
                sympyRep += coeff
            else:
                if fold_keys:
                    if nice_negatives:
                        key = tuple(latex(k) if k>=0 else (r'\bar{%s}' % latex(-k)) for k in key)
                    sympyRep += coeff*Ket(*key)
                else:
                    sympyRep += coeff*Ket(key)
        return sympyRep
        def as_bra(self):
            '''
            Give a representation of the qet  as  a  Bra  from
            sympy.physics.quantum.
            '''
            sympyRep = S(0)
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
        sympyRep = S(0)
        for key, coeff in self.dict.items():
            l = int(len(key)/2)
            if key == ():
                sympyRep += coeff
            else:
                sympyRep += coeff*(Bra(*key[:l])*Ket(*key[l:]))
        return sympyRep

    def as_symbol_sum(self):
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
        sympyRep = S(0)
        for key, op_and_coeff in self.dict.items():
            ops_and_coeffs = list(zip(op_and_coeff[::2],op_and_coeff[1::2]))
            for op, coeff in ops_and_coeffs:
                if key == ():
                    sympyRep += coeff
                else:
                    sympyRep += coeff*op(*key)#Function(op)(*key)
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
        return sqrt(norm2)

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

# def parse_math_expression(stringo):
#     '''Use to parse output from Mathematica'''
#     if 'Subscript' in stringo:
#       args = re.findall(r'\[(.*)\]',stringo)[0].replace('"','').replace(' ','').split(',')
#       if len(args[0]) > 1:
#         symb = '{%s}_{%s}' % tuple(args)
#       else:
#         symb = '%s_{%s}' % tuple(args)
#     elif 'Subsuperscript' in stringo:
#       args = re.findall(r'\[(.*)\]',stringo)[0].replace('"','').replace(' ','').split(',')
#       if len(args[0]) > 1:
#         symb = '{%s}_{%s}^{%s}' % tuple(args)
#       else:
#         symb = '%s_{%s}^{%s}' % tuple(args)
#     elif '_' in stringo:
#         symb = '{%s}_{%s}' % tuple(stringo.split('_'))
#     else:
#       symb = stringo
#     if '”' in stringo:
#       symb = symb.replace('”',"^{''}")
#     return symb

class ProductTable():
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
    """Class for group character tables"""
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
        self.direct_product_table()

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
        return qet.as_symbol_sum()

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
    """Class of all crystallographic point groups"""
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


###########################################################################
#################### Calculation of Surface Harmonics #####################

def SubSupSymbol(radix,ll,mm):
    '''
    Generates a symbol placeholder for the B coefficients in the crystal field potential.
    '''
    SubSupSym = symbols(r'{%s}_{%s}^{%s}' % (radix, str(ll), str(mm)))
    return SubSupSym

def SubSubSymbol(radix,ll,mm):
    '''
    Generates a symbol placeholder for the B coefficients in the crystal field potential.
    '''
    SubSubSym = symbols(r'{%s}_{{%s}{%s}}' % (radix, str(ll), str(mm)))
    return SubSubSym

def kronecker(i,j):
    return 0 if i!=j else 1

def Wigner_d(l, n, m, beta):
    k_min = max([0,m-n])
    k_max = min([l-n,l+m])
    Wig_d_prefact = sqrt((factorial(l+n)
                          *factorial(l+m)
                          *factorial(l-n)
                          *factorial(l-m)))
    Wig_d_summands = [((-S(1))**(k - m + n)
                      * cos(beta/2)**(2*l+m-n-2*k)
                      * sin(beta/2)**(2*k+n-m)
                      / factorial(l - n -k)
                      / factorial(l + m - k)
                      / factorial(k)
                      / factorial(k-m+n)
                      )
                      for k in range(k_min,k_max+1)]
    Wig_d = (Wig_d_prefact*sum(Wig_d_summands)).doit()
    return Wig_d

def Wigner_D(l, n, m, alpha, beta, gamma):
    args = (l, n, m, alpha, beta, gamma)
    if args in Wigner_D.values.keys():
      return Wigner_D.values[args]
    if beta == 0:
      Wig_D = exp(-I*m*alpha-I*m*gamma) * kronecker(n,m)
      if n == m:
        Wig_D = (cos(-m*alpha-m*gamma)+I*sin(-m*alpha-m*gamma))
      else:
        Wig_D = 0
    elif beta == pi:
      if n == -m:
        Wig_D = (-1)**l * (cos(-m*alpha + m*gamma)+I*sin(-m*alpha + m*gamma))
      else:
        Wig_D = 0
    else:
      Wig_D_0 = I**(abs(n)+n-abs(m)-m)
      Wig_D_1 = (cos(-n*gamma-m*alpha)+I*sin(-n*gamma-m*alpha)) * Wigner_d(l,n,m,beta)
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
                if sign(partner) == sign(chunk):
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
    '''This would be rotateHarmonic in the Mathematica code. It is used
    in the projection of the spherical  harmonics  to  create  symmetry
    adapted wavefunctions.
    '''
    Rf = Qet()
    for nn in range(-l,l+1):
        wigD = Wigner_D(l, m, nn, alpha, beta, gamma)
        if wigD != 0:
          Rf = Rf + Qet({(l,nn): wigD})
    return (S(detRot)**l) * Rf

def flatten_matrix(mah):
  ''' a convenience function
  to flatten a sympy matrix into a
  list of lists'''
  return [item for sublist in mah.tolist() for item in sublist]

def cg_eqns(group, Γ1_idx, γ1_idx, Γ2_idx, γ2_idx, Γ3_idx, γ3_idx):
  '''
 This function takes a group, three indices for three irreducible
 representations, and three indices for elements of the corresponding
 irreducible representation matrices. It returns a list of equations that
 correspond to TSK 2.31, with each element corresponding to one of the group
 generators. Here, to make the notation uniform, the sum index of the RHS has
 been changed to a gamma3' and the capital Gamma, to Γ3_idx.
 This function is used in the algorithm for computing the Clebsch-Gordan coeffs, that correspond to the problem of writing the basis functions of a product of two irreducible representations in terms of the basis functions of its factors.
  '''
  # grab the set of matrices for the irreducible representations of the generators
  irrep_matrices = group.IrrRepMatrices
  D1, D2, D3 = [irrep_matrices[idx] for idx in [Γ1_idx,Γ2_idx,Γ3_idx]]
  d1, d2, d3 = list(map(len,[D1[0],D2[0],D3[0]])) # this is how many basis vecs in each
  D1 = list(map(Matrix,D1))
  D2 = list(map(Matrix,D2))
  D3 = list(map(Matrix,D3))
  num_generators = len(D1)
  eqns = []
  for gen_idx in range(num_generators):
    lhs = S(0)
    for γ1p_idx in range(d1):
      for γ2p_idx in range(d2):
        chevron = '<%d,%d|%d,%d>' % (γ1p_idx, γ2p_idx, Γ3_idx, γ3_idx)
        chevron = Symbol(chevron)
        coef = D1[gen_idx][γ1_idx, γ1p_idx] * D2[gen_idx][γ2_idx, γ2p_idx]
        if coef != 0:
          lhs += coef*chevron
    rhs = S(0)
    for γ3p_idx in range(d3):
      chevron = '<%d,%d|%d,%d>' % (γ1_idx, γ2_idx, Γ3_idx, γ3p_idx)
      chevron = Symbol(chevron)
      coef = D3[gen_idx][γ3p_idx,γ3_idx]
      if coef != 0:
        rhs += coef * chevron
    eqn = lhs-rhs
    if eqn != 0:
      eqns.append(eqn)
  return eqns

def group_clebsh_gordan_coeffs(group, Γ1, Γ2):
  '''
  Given a group and string labels for two irreducible representations
  Γ1 and Γ2 this function calculates the  Clebsh-Gordan  coefficients
  used to span the basis functions of  their  product in terms of the
  basis functions of their factors.
  This function returns a tuple, the  first  element being  a  matrix
  of symbols which determine  to  which  CG  coefficient the elements
  given in the second element of the tuple correspond to. These  sym-
  bols are constructed thus
  <i1,i2|i3,i4>
  (i1 -> index for basis function in Γ1)
  (i2 -> index for basis function in Γ2)
  (i3 -> index for an irreducible representation Γ3 in the group)
  (i4 -> index for a basis function of Γ3)
  '''
  # find the corresponding indices
  Γ1_idx = group.IrrReps.index(Γ1)
  Γ2_idx = group.IrrReps.index(Γ2)

  # find the direct sum decomposition
  Γ3s_bmask = representation_product(group, Γ1_idx, Γ2_idx)

  # and the corresponding indices
  Γ3s_idx = [idx for idx, boo in enumerate(Γ3s_bmask) if boo]

  # also grab the labels for them
  Γ3s = [group.IrrReps[idx] for idx in Γ3s_idx]

  # construct the labels for the basis functions of all irreps
  all_basis_labels = get_components(group)

  # separate out the labels for the bases of Γ1, Γ2
  basis_fun_labels = get_components(group)
  Γ1_basis_labels = basis_fun_labels[Γ1_idx]
  Γ2_basis_labels = basis_fun_labels[Γ2_idx]

  # determine the shape of the CG matrix
  print("CG is a ({size},{size})".format(size=len(Γ1_basis_labels)*len(Γ2_basis_labels)))

  # then create all the linear constraints
  all_eqns = []
  for γ1_idx in range(len(Γ1_basis_labels)):
    for γ2_idx in range(len(Γ2_basis_labels)):
      for Γ3_idx in Γ3s_idx:
        Γ3_basis_labels = basis_fun_labels[Γ3_idx]
        for γ3_idx in range(len(Γ3_basis_labels)):
          cg_args = (group, Γ1_idx, γ1_idx, Γ2_idx, γ2_idx, Γ3_idx, γ3_idx)
          eqns = cg_eqns(*cg_args)
          if len(eqns) > 0:
            all_eqns.extend(eqns)
  # remove all evident redundancies
  all_eqns = list(set(all_eqns))

  # collect all the symbols included in all_eqns
  free_symbols = set()
  for eqn in all_eqns:
    free_symbols.update(eqn.free_symbols)
  free_symbols = list(free_symbols)

  # convert to matrix of coefficients
  coef_matrix = Matrix([[eqn.coeff(cg) for cg in free_symbols] for eqn in all_eqns])

  # and simplify using the rref
  rref = coef_matrix.rref()[0]

  # turn back to symbolic and solve
  better_eqns = [r for r in rref*Matrix(free_symbols) if r!=0]
  better_sol = solve(better_eqns, free_symbols)
  # construct the unitary matrix with all the CGs
  U = []
  for γ1_idx in range(len(Γ1_basis_labels)):
    for γ2_idx in range(len(Γ2_basis_labels)):
      row = []
      for Γ3_idx in Γ3s_idx:
        Γ3_basis_labels = basis_fun_labels[Γ3_idx]
        for γ3_idx in range(len(Γ3_basis_labels)):
          chevron = ('<%d,%d|%d,%d>' % (γ1_idx, γ2_idx, Γ3_idx, γ3_idx))
          chevron = Symbol(chevron)
          row.append(chevron)
      U.append(row)
  # replace with the current solution
  Usymbols = Matrix(U)
  U = Matrix(U).subs(better_sol)
  # build the unitary constraints
  unitary_constraints = U*U.T - eye(U.shape[0])
  # flatten and pick the nontrivial ones
  unitary_set = [f for f in flatten_matrix(unitary_constraints) if f!=0]
  # solve
  unitary_sol = solve(unitary_set)
  print("%d solutions found" % len(unitary_sol))
  # use one solution
  Usols = [U.subs(sol) for sol in unitary_sol]
  return Usymbols, Usols

def get_components(Group):
    '''
    This function takes a group and returns the a list of lists
    each with  string  labels  for  the basis  functions of its
    irreducible representations.
    '''
    Chi_0 = np.array(Group.CharacterTable)[:,0]
    Gamma_0 = Group.IrrReps

    componentList = []
    for comp in np.arange(len(Chi_0)):
        irrrep = Gamma_0[comp]
        switcher={
        1: [''.join(['a_{',irrrep,'}'])],
        2: [''.join(['u_{',irrrep,'}']),''.join(['v_{',irrrep,'}'])],
        3: [''.join(['x_{',irrrep,'}']),''.join(['y_{',irrrep,'}']),''.join(['z_{',irrrep,'}'])],
        }
        componentList.append(switcher.get(Chi_0[comp], "ERROR"))
    return componentList

def component_idx(Group):
    '''
    Given a group this function returns a a list of lists
    each with the  indices  that  correspond to the basis
    vectors of its irreducible representations.
    '''
    CompLen = list(map(np.shape, get_components(Group)))
    for idx in np.arange(len(CompLen)):
        ent = CompLen[idx]
        if ent == ():
            CompLen[idx] = 1
        else:
            CompLen[idx] = ent[0]

    AccComList = np.cumsum(CompLen)

    return [np.arange(x-1,y-1+0.1).astype(int).tolist() for x,y in zip((np.insert(AccComList[0:-1]+1,0,1)).astype(int),AccComList)]

def representation_product(Group, Gamma1, Gamma2, verbose = False):
    '''
    This function takes  a  group  (Group)  and  two  indices  (Gamma1, Gamma2)
    and returns a list of bools that indicate which irreducible representations
    compose the direct sum decomposition  of  the  irreducible  representations
    corresponding to the two given indices.
    If verbose=True, then the function prints a string that  shows  the  direct
    sum decomposition.
    '''
    if isinstance(Gamma1,str):
        Gamma1 = Group.IrrReps.index(Gamma1)
    if isinstance(Gamma2,str):
        Gamma2 = Group.IrrReps.index(Gamma2)

    Chi = Matrix(np.array(Group.CharacterTable[Gamma1])*np.array(Group.CharacterTable[Gamma2]))

    IRBool = np.array(list(linsolve((Matrix(Group.CharacterTable).T,Chi)).args[0])).astype(bool).tolist()

    if verbose == True:
        RepString = ' + '.join(np.array(Group.IrrReps)[IRBool])
        print('%s x %s : %s'%(Group.IrrReps[Gamma1],Group.IrrReps[Gamma2], RepString))
    return IRBool

def SymmetryAdaptedWF_old(group, l, m):
  '''
  This returns the proyection of Y_l^m
  on the trivial irreducible representation
  of the given group
  '''
  if isinstance(group,str):
      group = CPG.Groups[CPG.AllGroupLabels.index(group)]
  degree = 1
  # Order of the group which  is  equal  to
  # the number of the elements
  order = group.order
  SALC = Qet()
  # This sum is over all elements of the group
  for group_idx, group_op in enumerate(group.Elements):
    alpha, beta, gamma, detRot = group.ParameterTable[group_idx][:4]
    SALC += RYlm(l,m,alpha,beta,gamma,detRot)
  SALC = (S(1)/order)*SALC
  SALC = SALC.apply(lambda x,y : (x, simplify(y)))
  return SALC

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
  SALC = (S(1)/order)*SALC
  SALC = SALC.apply(lambda x,y : (x, simplify(y)))
  return SALC

def linearly_independent(vecs):
  '''given a list of vectors
  return the largest subset which
  of linearly independent ones
  and the indices that correspond
  to them in the original list
  '''
  matrix = Matrix(vecs).T
  good_ones = matrix.rref()[-1]
  return good_ones, [vecs[idx] for idx in good_ones]

def SymmetryAdaptedWFs(group, l, normalize=True, verbose=False, sympathize=True):
  '''For a given group and a given value of
  l, this returns a set of linearly independent
  symmetry adapted functions which are also real-valued.
  If the set that is found initially contains combinations that are
  not purely imaginary or pure real, then the assumption
  is made that this set contains single spherical
  harmonics, and then sums and differences between
  m and -m are given by doing this through the values
  of |m| for the functions with mixed character.'''

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
      WF = WF*(S(1)/norm)
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
        qp = qp*(S(1)/sqrt(2))
        qm = qm*(S(1)/sqrt(2))
      realWFs.append(qp)
      realWFs.append(qm)
    elif m%2 == 1:
      qp = Qet({(l,m): I}) + Qet({(l,-m): I})
      qm = Qet({(l,m): 1}) + Qet({(l,-m): -1})
      if normalize:
        qp = qp*(S(1)/sqrt(2))
        qm = qm*(S(1)/sqrt(2))
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
  lin_indep_vecs = list(map(list,GramSchmidt([Matrix(vec) for vec in lin_indep_vecs], normalize)))
  finalWFs = []
  if sympathize:
    better_vecs = []
    for vec in lin_indep_vecs:
      clear_elements = [abs(v) for v in vec if v!=0]
      if len(list(set(clear_elements))) == 1:
        better_vec = [0 if vl == 0 else sign(vl) for vl in vec]
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
            to_add = math.ceil( (max_cols - len(row_data))/2 )
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
    from sympy import sqrt
    THI = sqrt((2*l1+1)/(2*l3+1))*CG(l2,0,l3,0,l1,0)*CG(l2,m2,l3,m3,l1,m1)
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
    from sympy import Symbol, Matrix, simplify

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

    l1 = Symbol('l1')
    m1 = Symbol('m1')
    l3 = Symbol('l3')
    m3 = Symbol('m3')

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

    EigenSys = Matrix(CFP_Table).eigenvects()
    EigenVals = []
    EigenVecs = []
    for eRes in EigenSys:
        for deg in np.arange(eRes[1]):
            EigenVals.append(eRes[0])
            EigenVecs.append(list(eRes[2][deg]))

    Yarray = []
    for mm in np.arange(-orbital,orbital+0.1):
        Yarray.append(Ynm(2,int(mm),theta,phi))
    Yarray = Matrix(Yarray)
    EigenVecs = Matrix(EigenVecs)

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
            colsimp = simplify(col)
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

def Yrot(l,m,theta,phi):
    return RYlm(l, m, alpha, beta, gamma, detRot)

if os.path.exists(os.path.join(module_dir,'data','CPGs.pkl')):
    print("Reloading...")
    CPGs = pickle.load(open(os.path.join(module_dir,'data','CPGs.pkl'),'rb'))
else:
    print("Regenerating...")
    CPGs = CPGroups(group_data)
