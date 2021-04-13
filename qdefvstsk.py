
from sympy import Function, S, eye, solve
from sympy import symbols, Matrix, sqrt, sign, latex
from sympy.physics.quantum import TensorProduct
import pickle
from itertools import product, combinations
from collections import OrderedDict
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.wigner import clebsch_gordan as ClebschG
from IPython.display import Math

u, v, zeta, eta, xi, x, y, z = symbols("u v zeta eta xi x y z")
e1, e2 = symbols("e1 e2")
alpha, beta, gamma = symbols("alpha beta gamma")
Oireps = pickle.load(open('/Users/juan/Google Drive/Zia Lab/Log/Data/Oireps.pkl','rb'))

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

Odict = pickle.load(open('/Users/juan/Google Drive/Zia Lab/Log/Data/O_table.pkl','rb'))
products = Odict['products']
Otable = Odict['table']


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

class Qet():
  '''
  Scalars may be added to a braket by using
  the empty type as a key.
  Up to you to make sure that product of Qets
  make sense.
  '''
  def __init__(self,bits):
    assert type(bits) == dict, 'Input must be a dictionary.'
    self.dict = {k: v for k,v in bits.items() if v!=0}

  def __add__(self, other):
    new_dict = dict(self.dict)
    for key, coeff in other.dict.items():
      if key in new_dict.keys():
        new_dict[key] += coeff
      else:
        new_dict[key] = coeff
    return Qet(new_dict)

  def __mul__(self, multiplier):
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
    new_dict = dict(self.dict)
    for key, coeff in new_dict.items():
      new_dict[key] = multiplier*(coeff)
      return Qet(new_dict)


  def dual(self):
    new_dict = dict(self.dict)
    for key, coeff in new_dict.items():
      new_dict[key] = conjugate(coeff)
    return Qet(new_dict)

  def as_ket(self):
    sympyRep = S(0)
    for key, coeff in self.dict.items():
      if key == ():
        sympyRep += coeff
      else:
        sympyRep += coeff*Ket(key)
    return sympyRep

  def as_bra(self):
    sympyRep = S(0)
    for key, coeff in self.dict.items():
      if key == ():
        sympyRep += coeff
      else:
        sympyRep += coeff*Bra(*key)
    return sympyRep

  def as_braket(self):
    sympyRep = S(0)
    for key, coeff in self.dict.items():
      l = int(len(key)/2)
      if key == ():
        sympyRep += coeff
      else:
        sympyRep += coeff*(Bra(*key[:l])*Ket(*key[l:]))
    return sympyRep

  def as_c_number_with_fun(self):
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
        if new_coeff !=0 :
          if new_key not in new_dict.keys():
            new_dict[new_key] = (new_coeff)
          else:
            new_dict[new_key] += (new_coeff)
    return Qet(new_dict)

  def norm(self):
    norm2 = 0
    for key, coeff in self.dict.items():
      norm2 += abs(coeff)**2
    return sqrt(norm2)

  def symmetrize(self):
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

  def asymmetrize(self):
    new_dict = dict()
    for key, coeff in self.dict.items():
      rkey = key[::-1]
      if rkey in new_dict.keys():
        if isinstance(coeff,tuple):
          new_dict[rkey] += (coeff[0],-coeff[1])
      else:
        if key in new_dict.keys():
          new_dict[key] += coeff
        else:
          new_dict[key] = coeff
    return Qet(new_dict)

  def __repr__(self):
    return str(self.dict)

def qet_sum(qets):
  sqet = Qet({})
  for qet in qets:
    sqet = sqet + qet
  return sqet

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
#             printout += wowee + r'\\'
            # save in results dictionary
            kay = (irep3, s3tot, one_s3z, one_gamma)
            full_results[kay] = rket
            if printem:
              display(Math(wowee))
#             printouts[kay] = wowee
  if returnthem:
    return full_results
