from sympy import Function, S, eye, solve, pi
from sympy import symbols, Matrix, sqrt, sign, latex, Rational
from sympy.physics.quantum import TensorProduct
import pickle
from itertools import product, combinations
from collections import OrderedDict
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.wigner import clebsch_gordan as ClebschG
from IPython.display import Math, display
import matplotlib.pyplot as plt
import numpy as np
from sympy.vector import CoordSys3D

u, v, zeta, eta, xi, x, y, z = symbols("u v zeta eta xi x y z")
e1, e2 = symbols("e1 e2")
alpha, beta, gamma = symbols("alpha beta gamma")
Oireps = pickle.load(open('/Users/juan/Google Drive/Zia Lab/Log/Data/Oireps.pkl','rb'))
xyz = CoordSys3D('N')
x,y,z,r = symbols('x y z r')

symbols_dictionary = {'E': [u, v],
                     'T1': [alpha, beta, gamma], #[x, y, z],
                     'T2': [xi, eta, zeta],
                     'A1': [e1],
                     'A2': [e2]}

totals = '''a1 a1 a1
a1 a2 a2
a1 e e
a1 t1 t1
a1 t2 t2
a2 a1 a2
a2 a2 a1
a2 e e
a2 t1 t2
a2 t2 t1
e a1 e
e a2 e
e e a1 a2 e
e t1 t1 t2
e t2 t1 t2
t1 a1 t1
t1 a2 t2
t1 e t1 t2
t1 t1 a1 t1 t2 e
t1 t2 a2 e t1 t2
t2 a1 t2
t2 a2 t1
t2 e t1 t2
t2 t1 a2 t1 t2 e
t2 t2 a1 e t1 t2'''.upper()
products = OrderedDict()
for line in totals.split('\n'):
  r1, r2, *targets = line.split()
  products[(r1,r2)] = targets

protectex = {
    zeta: r'{\zeta} ',
    eta: r'{\eta} ',
    xi: r'{\xi} '}

# hard coded ireps and chartable for the generators of O
# for each irreducible representation one matrix is given
# for each of the group classes

Ogroup = OrderedDict()
Ogroup['ireps'] = OrderedDict()
Ogroup['ireps']['A1'] = OrderedDict(
  [('E', [[1]]),
  ('C_{4z}', [[1]]),
  ('C_{4z}^2', [[1]]),
  ('C_{3xyz}', [[1]]),
  ('C_{2xy}', [[1]])]
)
Ogroup['ireps']['A2'] = OrderedDict(
  [('E', [[1]]),
  ('C_{4z}', [[-1]]),
  ('C_{4z}^2', [[1]]),
  ('C_{3xyz}', [[1]]),
  ('C_{2xy}', [[-1]])]
)
Ogroup['ireps']['E'] = OrderedDict(
  [('E', [[1,0],[0,1]]),
  ('C_{4z}', [[1,0],[0,-1]]),
  ('C_{4z}^2', [[1,0],[0,1]]),
  ('C_{3xyz}', [[-Rational(1,2), -sqrt(3)*Rational(1,2)],
              [sqrt(3)*Rational(1,2), -Rational(1,2)]]),
  ('C_{2xy}', [[1,0],[0,-1]])]
)
Ogroup['ireps']['T2'] = OrderedDict(
  [('E', [[1,0,0],[0,1,0],[0,0,1]]),
  ('C_{4z}', [[0,1,0],[-1,0,0],[0,0,-1]]),
  ('C_{4z}^2', [[-1,0,0],[0,-1,0],[0,0,1]]),
  ('C_{3xyz}', [[0,0,1],[1,0,0],[0,1,0]]),
  ('C_{2xy}', [[0,-1,0],[-1,0,0],[0,0,1]])]
)
Ogroup['ireps']['T1'] = OrderedDict(
  [('E', [[1,0,0],[0,1,0],[0,0,1]]),
  ('C_{4z}', [[0,-1,0],[1,0,0],[0,0,1]]),
  ('C_{4z}^2', [[-1,0,0],[0,-1,0],[0,0,1]]),
  ('C_{3xyz}', [[0,0,1],[1,0,0],[0,1,0]]),
  ('C_{2xy}', [[0,1,0],[1,0,0],[0,0,-1]])]
)
Ogroup['chartable'] = Matrix([[1,1,1,1,1],
              [1,-1,1,1,-1],
              [2,0,2,-1,0],
              [3,1,-1,0,-1],
              [3,-1,-1,0,1]])

Ogroup['class_sizes'] = {'E': 1,
               'C_{4z}': 6,
               'C_{4z}^2': 3,
               'C_{3xyz}': 8,
               'C_{2xy}': 6}

Odict = pickle.load(open('/Users/juan/Google Drive/Zia Lab/Log/Data/O_table.pkl','rb'))
products = Odict['products'] # what ireps result from a product of two
Otable = Odict['table']

def rot_matrix(rot_params):
  '''
  This returns a matrix  for  a given set  of  rotation
  parameters.
  rot_params is a tuple with rot_params[0] the rotation
  angle, and rot_params[1] a sympy vector for the rotation
  axis.
  The rotation axis needs not be a unit vector.

  e.g rot_matrix((pi/2, xyz.x + xyz.y)) returns the rotation matrix
  for a 90 degree rotation about the diagonal in the x,y plane
  '''
  A = xyz.orient_new_axis('A', *rot_params)
  return xyz.rotation_matrix(A)

def compute_matrix(group_op):
  full_matrix = []
  for m in [-2,-1,0,1,2]:
    subs = (rot_matrix(group_ops[group_op]).inv().dot([x,y,z])) # this determined the coord transform
    rot_op = YC2m[m].subs([(x,subs[0]),(y,subs[1]), (z,subs[2])], simultaneous=True)
    rot_op = rot_op.subs(r,1).subs(x, sin(theta)*cos(phi)).subs(y, sin(theta)*sin(phi)).subs(z, cos(theta))
    proyection = []
    for m2 in [-2,-1,0,1,2]:
      integrand = rot_op * (YS2mC[m2]) * sin(theta)
      integral2 = integrate(integrand, (phi,0,2*pi))
      integral = simplify(integrate(integral2,(theta,0,pi)))
      proyection.append(integral)
    full_matrix.append(proyection)
  return (group_op,subs,full_matrix)

def direct_product_table(ireps):
    '''for a given list of ireps
    return how their direct product
    separates as a direct sum'''
    chars = []
    for gclass in Ogroup['ireps'][ireps[0]]:
      mats = [Matrix(Ogroup['ireps'][ir][gclass]) for ir in ireps]
      tprod = TensorProduct(*mats)
      chars.append(tprod.trace())
    chars = Matrix(chars)
    partition = (Ogroup['chartable'].T)**(-1)*chars
    qet = Qet()
    for element, ir in zip(partition,['A1','A2','E','T1','T2']):
        qet = qet + Qet({ir:element})
    return qet

def direct_product_table_v2(ireps):
    '''for a given list of ireps
    return how their direct product
    separates as a direct sum'''
    ir_labels = ['A1','A2','E','T1','T2']
    chars = []
    for gclass in Ogroup['ireps'][ireps[0]]:
      chars = [Ogroup['chartable'][ir_labelds.index(ir)] for ir in ireps]
      tprod = TensorProduct(*mats)
      chars.append(tprod.trace())
    chars = Matrix(chars)
    partition = (Ogroup['chartable'].T)**(-1)*chars
    qet = Qet()
    for element, ir in zip(partition,['A1','A2','E','T1','T2']):
        qet = qet + Qet({ir:element})
    return qet

def match_spin(key,val):
  '''
  This nulls brakets with no matching spin values,
  after this spin is considered irrelevant.
  (TSK 2.61)
  '''
  m1, m2, m1p, m2p = key[::2]
  g1, g2, g1p, g2p = key[1::2]
  if dirac((m1,m2),(m1p,m2p)):
    return {(g1,g2,g1p,g2p): val}
  else:
    return {():0}

def interaction_integrals(factor_ireps, ireps, verbose=False):
  '''
  Given  the  labels  for  four  irreducible   representations
  (factor_ireps) and a dictionary (ireps)  with  matrices  and
  basis symbols for operations in  the  symmetry  group,  this
  function determines which two electron spherical brakets are
  linearly dependent and  what  is  their  dependence  to  the
  linearly independent ones.
  In the braket <γ1γ2|R|γ3γ4>:
   γ1 -> all_ireps[0]
   γ2 -> all_ireps[1]
   γ3 -> all_ireps[2]
   γ4 -> all_ireps[3].
  The function returns a dictionary that gives  the  dependent
  brakets in terms of the independent ones. As of now this has
  been tested for the cubic group O.
  '''

  # get the names for the symmetry operations
  ops = list(list(ireps.values())[0]['matrices'].keys())
  # the identity op won't provide any information
  ops.remove('E')

  # initialize an empty matrix
  bigMatrix = Matrix([])
  # collect the symbols from the all the ireps
  syms = [ireps[ir]['basis'] for ir in factor_ireps]
  # collect the matrices
  mats = [ireps[ir]['matrices'] for ir in factor_ireps]
  # compute all symbols for four symbol brakets
  states = list(product(*syms))
  # query the size of each irep
  ls = [len(ireps[ir]['basis']) for ir in factor_ireps]
  # determine how many four symbol brakets there are
  hilbert_size = len(states)

  irep1, irep2, irep3, irep4 = factor_ireps

  # how the four symbol brakets are dependent on each
  # other may be  found  by  vertically stacking  the
  # tensor products of the matrices that correspond to
  # all of the group operations, subtracting the identity
  # from this blocks, and finding the resulting reduced
  # echelon form

  for op in ops:
    mat1, mat2, mat3, mat4 = [Matrix(mat[op]) for mat in mats]
    g = (TensorProduct(mat1,mat2,mat3,mat4) - eye(hilbert_size))
    bigMatrix = Matrix.vstack(bigMatrix,g)

  # If basis functions are real, and there are repeated ireps,
  # then there is also a symmetry between exchange of symbols

  permMatrix = Matrix([])
  for basis_vec in states:
    l0, l1, l2, l3 = basis_vec
    q1 = (l0,l1,l2,l3)
    q2 = (l2,l1,l0,l3)
    q3 = (l0,l3,l2,l1)
    q4 = (l2,l3,l0,l1)

    if irep1 == irep3:
      row = [0]*hilbert_size
      row[states.index(q1)] = 1
      row[states.index(q2)] = -1
      if row.count(1) == 1:
        row = Matrix([row])
        permMatrix = Matrix.vstack(permMatrix,row)

    if irep2 == irep4:
      row = [0]*hilbert_size
      row[states.index(q1)] = 1
      row[states.index(q3)] = -1
      if row.count(1) == 1:
        row = Matrix([row])
        permMatrix = Matrix.vstack(permMatrix,row)

    if (irep1 == irep3) and (irep2 == irep4):
      row = [0]*hilbert_size
      row[states.index(q1)] = 1
      row[states.index(q4)] = -1
      if row.count(1) == 1:
        row = Matrix([row])
        permMatrix = Matrix.vstack(permMatrix,row)
  if len(permMatrix)>0:
    bigMatrix = Matrix.vstack(bigMatrix,permMatrix)

  protectex = {
      zeta: r'{\zeta} ',
      eta: r'{\eta} ',
      xi: r'{\xi} '}
  variables = list(map(lambda x: Qet({x:1}),states))
  rref = bigMatrix.rref()
  qets = []
  for row in rref[0].rowspace():
    qet = Qet({():0})
    for bv, c  in zip(states,row):
      qet = qet + Qet({(bv):c})
    qets.append(qet)
  sol_eqns = ([qet.as_braket() for qet in qets])
  sol_vars = ([qet.as_braket() for qet in variables])
  subs = solve(sol_eqns,sol_vars)
  indep_ints = (len(states)-len(rref[0].rowspace()))
  if verbose:
    print("There are %d independent integrals." % indep_ints)
    for key,val in subs.items():
      if val!= 0:
        display(Math('%s = %s' % (latex(key,symbol_names=protectex), latex(val,symbol_names=protectex))))
  return subs

def det_braket_expand(key, val):
  '''
  This converts a braket
  of determinat states into
  regular brakets.
  (TSK 2.59)
  '''
  returndict = dict()
  m1, m2, m1p, m2p = key[::2]
  g1, g2, g1p, g2p = key[1::2]
  returndict[(m1,g1,m2,g2,m1p,g1p,m2p,g2p)] = val
  if (m1,g1,m2,g2,m1p,g1p,m2p,g2p) == (m1,g1,m2,g2,m2p,g2p,m1p,g1p):
    returndict[(m1,g1,m2,g2,m2p,g2p,m1p,g1p)] += -val
  else:
    returndict[(m1,g1,m2,g2,m2p,g2p,m1p,g1p)] = -val
  return returndict

J = Function("J")
K = Function("K")

def interactionbraket(key, val):
  '''
  This takes a braket
  and writes it up in terms
  of J and K integrals
  (TSK 2.61)
  '''
  returndict = dict()
  g1, g2, g1p, g2p = key
  if (g1 == g1p) and (g2 == g2p):
    returndict[(g1,g2)] = val
  else:
    returndict[(g1,g1p)] = val
  return returndict

def dirac(p1, p2):
  if p1 == p2:
    return 1
  else:
    return 0

def compute_basis(kets,more_kets=[]):
  '''
  returns a sorted list  with elements
  equal to the quantum numbers of  the
  provided iterable of qets  (list  or
  np.array) if the second  argument is
  provided the iterables of  qets  are
  concatenated into a single list
  '''
  if len(more_kets) > 0:
    if isinstance(kets, list):
       kets = kets + more_kets
    elif isinstance(kets, np.ndarray):
       kets = list(kets) + list(more_kets)
  basis = []
  for ket in kets:
   basis.extend(ket.basis())
  basis = list(set(basis))
  basis = list(sorted(basis))
  return basis

class Qet():
  '''
  A Qet is a dictionary of keys and values. Keys
  correspond to tuples of quantum numbers or symbols
  and the values correspond to the accompanying
  coefficients.
  Scalars may be added by  using  an  empty tuple
  as a key.
  A qet may be multiplied by a scalar,  in  which
  case all the coefficients are multiplied by it,
  It  may  also  be  multiplied by  another  qet,
  in which case quantum numbers are  concatenated
  and coefficients multiplied accordingly.

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
    '''given an ordered basis  return   a
    list with the coefficients of the qet
    in that basis'''
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
    '''if multiplier is another
    qet, then concatenate the dict and
    multiply coefficients, if multiplier is
    something else then try to multiply
    the qet coefficients by the given multiplier'''
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
    '''give a representation of the qet
    as a Ket from sympy.physics.quantum
    fold_keys = True removes unnecessary parentheses
    and nice_negatives = True assumes all numeric keys
    and presents negative values with a bar on top'''
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
    '''give a representation of the qet
    as a Bra from sympy.physics.quantum'''
    sympyRep = S(0)
    for key, coeff in self.dict.items():
      if key == ():
        sympyRep += coeff
      else:
        sympyRep += coeff*Bra(*key)
    return sympyRep

  def as_braket(self):
    '''give a representation of the qet
    as a Bra*Ket the dict of the qet are
    assumed to split half for the bra, and
    other half for the ket.'''
    sympyRep = S(0)
    for key, coeff in self.dict.items():
      l = int(len(key)/2)
      if key == ():
        sympyRep += coeff
      else:
        sympyRep += coeff*(Bra(*key[:l])*Ket(*key[l:]))
    return sympyRep

  def as_symbol_sum(self):
    tot = S(0)
    for k, v in self.dict.items():
      tot += v*symbols(k)
    return tot

  def as_c_number_with_fun(self):
    '''the coefficients of a qet can be tuples
    and if the first element is a function then
    this method can be used to apply that function
    to the dict and multiply that result by the
    coefficient, which is assumed to be the second
    element of the tuple'''
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
    '''this method can be used to apply a function to a qet
    the provided function f must take as arguments a single
    pair of qnum and coeff and return a dictionary or a
    (qnum, coeff) tuple'''
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

def two_electron_configs(irep1, irep2, returnthem = False, printem = False):
  '''
    Give two irreducible representations irep1 and irep2
    find the expression for the resulting  wavefunctions
    in terms of determinant states taken  from the bases
    for the irreducible reps.
    What is  returned  is a  dictionary  whose  keys are
    electron  configurations  and  whose  values are Qet
    objects  that  must  be  interpreted  as determinant
    states.
  '''
  s1 = S(1)/2
  s2 = S(1)/2
  irep3s = products[(irep1,irep2)]
  R1symbols = symbols_dictionary[irep1]
  R2symbols = symbols_dictionary[irep2]
  basis_kets_dict = {'E':'e','T1':'t_1','T2':'t_2','A1':'a_1','A2':'a_2'}
  full_results = {}
  printouts = {}
  counter = 0

  for irep3 in irep3s:
    R3symbols = symbols_dictionary[irep3]
    for s3tot in {'T1': [0,1], 'T2': [0,1], 'E': [0], 'A1': [0], 'A2': [0,1]}[irep3]: # loop through total angular momenta
      for one_gamma in R3symbols: # loop through the various basis states for the irrep
        all_kets = {}
        for one_s3z in {0:[0],1:[1,0,-1]}[s3tot]: # loop through z eigenvalues of s3tot
          product_state = {}
          for m1,m2,gamma1,gamma2 in product((-1,1), (-1,1), R1symbols, R2symbols): # this loop is to create a single state
            # print(irep1, irep2, irep3, gamma1)
            gCG = (Otable[(irep1, irep2)]
                   [irep3]
                   [gamma1]
                   [R2symbols.index(gamma2),R3symbols.index(one_gamma)])
            sCG = ClebschG(s1, s2, s3tot, S(m1)/2, S(m2)/2, one_s3z)
            aket = ((S(m1)/2, gamma1, S(m2)/2, gamma2))
            if aket not in product_state.keys():
              product_state[aket] = 0
            product_state[aket] += sCG * gCG

          ket_dict = {}
          if irep1 == irep2:
            standard_order = []
            for p1,p2 in list(combinations(list(product([-S(1)/2,S(1)/2],R1symbols)), 2)):
              standard_order.append((*p1,*p2))
          else:
            standard_order = []
          for ket, coeff in product_state.items():
            ket_args = ket
            symbol_combo = ket_args
            if symbol_combo not in standard_order:
              coeff = -coeff
              ket_args_i = ket_args[-2:] + ket_args[:2]
              new_ket_key = (ket_args_i[0],ket_args_i[1],ket_args_i[2],ket_args_i[3])
            else:
              new_ket_key = (ket_args[0],ket_args[1],ket_args[2],ket_args[3])
            if new_ket_key not in ket_dict.keys():
              ket_dict[new_ket_key] = 0
            ket_dict[new_ket_key] += coeff
          coeffs_norm = Matrix(list(ket_dict.values())).norm()

          # make simplifications and create det states

          nice_ket = {}
          neg_sign_counter = 0
          non_zero_coeffs = 0
          for ket, coeff in ket_dict.items():
            ket_args = ket
            # make nice chevrons for the determinant states
            if sign(ket_args[0]) < 0:
              chevron_1 = r'\bar{%s}' % latex(ket_args[1])
            else:
              chevron_1 = r'{%s}' % latex(ket_args[1])
            if sign(ket_args[2]) < 0:
              chevron_2 = r'\bar{%s}' % latex(ket_args[3])
            else:
              chevron_2 = r'{%s}' % latex(ket_args[3])
            # if the chevrons are the same, order the positive one on the first slot
            # if a permutation is necessary change the sign of the coefficient
            if ket_args[1] == ket_args[3]: #same symbols
              if sign(ket_args[0]) > 0:
                chevron_1, chevron_2 = chevron_1, chevron_2
              else:
                chevron_1, chevron_2 = chevron_1, chevron_2
                chevron_2, chevron_1 = chevron_1, chevron_2
                coeff = -coeff
            if chevron_1 == chevron_2: # same symbols and spin
              continue
            ket_chunk = (chevron_1, chevron_2)
            if ket_chunk not in nice_ket.keys():
              nice_ket[ket_chunk] = 0
            nice_ket[ket_chunk] += coeff
            if coeff < 0:
              neg_sign_counter = neg_sign_counter + 1
            if coeff != 0:
              non_zero_coeffs += 1

          # now put together the full ket with the coefficients and basis kets
          # find the norm (it assumes that all necessary identities have been enforced)
          rket = Qet(nice_ket) # *(1/normalizer)
          rketNorm = rket.norm()
          # grab the symbols for the current single electron states
          e1_symbol = basis_kets_dict[irep1]
          e2_symbol = basis_kets_dict[irep2]
          if rketNorm != 0:
            rket = rket*(S(1)/rketNorm)
            if neg_sign_counter > non_zero_coeffs/2:
              rket = rket*(-S(1))
            else:
              rket = rket*(S(1))
            lhs = latex(Ket((("{%s} \cdot {%s}" % ((e1_symbol), (e2_symbol))),"{}^%s{%s}"%({0:'1',1:'3'}[s3tot],{'E':'E','T1':'T_1','T2':'T_2','A1':'A_1','A2':'A_2'}[irep3]),"M="+latex(one_s3z, symbol_names=protectex),one_gamma)), symbol_names=protectex)
            rhs = latex(rket.as_ket(), symbol_names=protectex).replace(r'\right\rangle',r'\right|').replace(r'\left(','').replace(r'\right)','').replace(r', ','').replace(r'\ ','')
            wowee = "%s = %s" % (lhs, rhs)
            kay = (irep3, s3tot, one_s3z, one_gamma)
            full_results[kay] = rket
            if printem:
              display(Math(wowee))
  if returnthem:
    return full_results

def main():
    print("qdef")

if __name__ == "__main__":
    main()
