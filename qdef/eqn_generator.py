#!/usr/bin/env python3

from misc import latex_eqn_to_png

eqns = {'hamiltonian': r'''\[\hat{H} = \sum_i^N \frac{\mathbf{\hat{p}}^2}{2m_e}
   + \sum_i^N V_\textrm{r}(\vec{r_i})
   + \sum_{i>j}^N \frac{e^2}{|\vec{r}_j-\vec{r}_i|}
   + \sum_i^N V_\textrm{CF}(\vec{r_i})
   + \alpha_T L (L+1)
   + \sum_{i}^N \zeta(r) \hat{l_i}\cdot\hat{s_i}\]'''}

for eqn_name, latex_code in eqns.items():
    latex_eqn_to_png(latex_code,timed=False, figname=eqn_name)
