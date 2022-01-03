#!/usr/bin/env python3

import sympy as sp
from itertools import product

def SlaterFddSTK(k, κ = sp.Symbol('\\kappa')):
    '''
    For a d-orbital in a hydrogenic atom 

        R_d(r) = N r^2 exp(-κ r)
        with N = sqrt((2κ)**7/6!). 
    
    Using these, this function returns the values of the
    Slater integral F^k(dd).

    Parameters
    ----------
    k : [int]
        The superscript for the corresponding Slater integral.
    κ : [sp.Symbol or number] dimension is of inverse  length,
        and corresponds to the exponential decay rate of   the
        wave function. 

    Returns
    -------
    Fk : [sympy expression]
        The corresponding Slater integral in terms of κ (kappa).

    Reference
    ---------
    STK eqn (5.3)
    '''    
    s0 = 4 * sp.factorial(6 + k) / sp.factorial(6)**2
    s10 = sp.factorial(5 - k)
    s11 = sum([sp.factorial(12-n) / 2**(13-n) / sp.factorial(7+k-n) for n in range(1,8+k)]) 
    s1 = s10 - s11
    Fk = κ * s0 * s1
    return Fk

def RkSlater_A(n, l, s):
    '''
    Used in RkSlater
    '''
    return ((-1)**s * (sp.S(2) / n)**s
            / sp.factorial(n - l - s - 1) / sp.factorial(2 * l + s + 1) / sp.factorial(s) 
            ) 

def RkSlater_I(s, s1, s2, s3, na, la, nb, lb, nc, lc, nd, ld, k):
    '''
    Used in RkSlater
    '''
    p = lb + ld + 2 + k + s1 + s3 
    q = la + lc + 1 - k + s  + s2 
    x = la + lc + 2 + k + s  + s2 
    y = lb + ld + 1 - k + s1 + s3 
    α = (na + nc) / (na * nc) 
    β = (nb + nd) / (nb * nd) 
    r00 = (sp.factorial(q) / α**(q+1) / β**(p+1)) *  (β / (α + β))**(p+1) 
    r01 = sum([sp.factorial(p+r) / sp.factorial(r) * (α / (α + β))**r for r in range(0,q+1)]) 
    r0 = r00 * r01 
    r10 = sp.factorial(y) / α**(x+1) / β**(y+1)   *    (α / (α + β))**(x+1) 
    r11 = sum([sp.factorial(x+r) / sp.factorial(r) * (β / (α + β))**r for r in range(0,y+1)]) 
    r1 = r10 * r11
    return r0 + r1 

def RkSlater(input_params):
    '''
    Provides the hydrogenic Slater radial integral

    R^{(k)}(ab,cd) = \int_0^\infty\int_0^\infty 
                    \frac{r_<^{k}}{r_>^{k+1}} 
                    R_{na,la}(r_1) R_{nb,lb}(r_2) 
                    R_{nc,lc}(r_1) R_{nd,ld}(r_2)
                    r_1^2 r_2^2 dr_1 dr_2
    
    Where a = (na,la), b = (nb, lb), c = (nc, lc), d = (nd, ld).

    Where R_{nl} = \sqrt{\left(\frac{2}{na}\right)^3 
                   \frac{(n-l-1)!}{2n (n+1)!}}
                   e^{-\rho/2} \rho^l L_{n-l-1}^{2l+1}(\frac{2r}{na})

    The  dimensions  of  this  integral are of inverse length, and what is
    provided  by  this  function  is  in  terms  of atomic units, that is,
    setting a=1.

    Parameters
    ----------
    inputparams (str, or tuple)
        If  given  as a string the format needs to be 'k na la nb lb nc lc
        nd  ld'  where  all  la,  lb,  lc,  ld  are given in spectroscopic
        notation.

        For  example  to  obtain  R^0(2s2s2s1s)  the input string would be
        '02s2s2s1s'.

        For  legibility  any  spaces can be added, for example '0 2s 2s 2s
        1s' is equivalent to '02s2s2s1s'.

        If  inputparams  is  given  as  a  tuple  then  it needs to have 9
        elements:

            ( k [int], na[int], ls[int], nb [int] ... , nd[int], ld[int])

    Returns
    -------
    R^(k)(ab,cd) as a sympy symbolic expression.

    References
    ----------
    Butler,  Minchin,  and  Wybourne,  "Tables of Hydrogenic Slater Radial
    Integrals." NOTE: A(abcd) was missing a few sqrt factorials.
    '''

    if isinstance(input_params, str):
        inputstring = input_params.replace(' ','')
        k, na, la, nb, lb, nc, lc, nd, ld = list(inputstring)
        la, lb, lc, ld = [{'s':0,'p':1,'d':2,'f':3, 'g':4}[x] for x in [la,lb,lc,ld]]
        k, na, nb, nc, nd = list(map(int, [k, na, nb, nc, nd]))
        k, na, la, nb, lb, nc, lc, nd, ld = list(map(sp.S, [k, na, la, nb, lb, nc, lc, nd, ld]))
        input_params = k, na, la, nb, lb, nc, lc, nd, ld
    else:
        k, na, la, nb, lb, nc, lc, nd, ld = input_params
    
    if input_params in RkSlater.cache:
        return RkSlater.cache[input_params]

    scaler = (sp.S(2)**4 / (na * nb * nc * nd)**2
                * sp.sqrt((sp.factorial(na - la -1)
                         * sp.factorial(nb - lb -1)
                         * sp.factorial(nc - lc -1)
                         * sp.factorial(nd - ld -1))
                         * sp.factorial(la + na) # added
                         * sp.factorial(lb + nb) # added
                         * sp.factorial(lc + nc) # added
                         * sp.factorial(ld + nd) # added
                         ) 
                * (sp.S(2) / na)**la 
                * (sp.S(2) / nb)**lb 
                * (sp.S(2) / nc)**lc 
                * (sp.S(2) / nd)**ld) 

    iterator =  product(range(0, na-la),
                        range(0, nb-lb),
                        range(0, nc-lc),
                        range(0, nd-ld)
                       )
    summands = [( RkSlater_A(na,la,s) 
                * RkSlater_A(nb,lb,s1)
                * RkSlater_A(nc,lc,s2)
                * RkSlater_A(nd,ld,s3)
                * RkSlater_I(s,s1,s2,s3,na,la,nb,lb,nc,lc,nd,ld,k)) for s,s1,s2,s3 in iterator]
    Rkabcd = scaler * sum(summands)
    RkSlater.cache[input_params] = Rkabcd
    return Rkabcd
RkSlater.cache = {}

def SlaterFdd(k, n = 3, κ = sp.Symbol('\\kappa')):
    '''
    Slater radial integral for nd-orbitals, using RkSlater.

    Parameters
    ----------
    k : (int)
        The superscript for the corresponding Slater integral.
    n : (int)
        The shell to which the atomic orbital belongs to.
    κ : (sp.Symbol or number) represents the reciprocal of the
        inverse of the unit of length for this hydrogenic model
        for an ion. What is oftem most important is the ratio,
        between two of these, so this factor is often irrelevant,
        given that it's shared between different integrals.

    Returns
    -------
    SFdd : [sympy expression]
        The corresponding Slater integral in terms of κ (kappa).

    '''    
    SFdd = RkSlater('{k}{n}d{n}d{n}d{n}d'.format(k=k, n=n)) * n * κ
    return SFdd