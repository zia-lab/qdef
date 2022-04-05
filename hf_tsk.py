#!/usr/bin/env python3

from qdef import *
import sympy as sp
import pandas as pd
from matrixgoodies import *
from uncertainties import ufloat
from misc import *
from collections import Counter
import h5py
import multiprocessing
from joblib import Parallel, delayed
import os

info = '''This script calculates the Tanabe-Sugano diagrams
for the different transition metal ions using Hartree-Fock
values for the Racah parameters and for spin-orbit coupling.'''

l = 2
hf_free_ions = pd.read_pickle('./data/brik_ma_cowan.pkl')
# hf_free_ions = hf_free_ions[hf_free_ions['Element'] == 'Cr']
# hf_free_ions.reset_index(inplace=True)
Dqs = np.linspace(-6,6,481)
Dq = sp.Symbol('Dq')
num_cores = multiprocessing.cpu_count()
disambiguate = False
data_folder = './data/eigen_ions_hr/'
energy_h5_file = './data/tsk_diag_withSO_HF-test-final-highres.h5'

def progress(total):
    num_done = len([f for f in os.listdir(data_folder) if 'h5' in f])
    print("%.1f" % (num_done/total*100))

def ion_solver(B, C, ζ, num_electrons, element, charge):
    if any(list(map(np.isnan, [B, C, ζ]))):
        return None
    else:
        F2 = 49*B + 7*C
        F4 = 63*C/5
        eigen_fname = os.path.join(data_folder,'%s-%d.h5' % (element, charge))
        ham = hamiltonian_CF_CR_SO_TO(num_electrons, 'O', l, False, True)
        matrixmah = ham[0].subs({sp.Symbol('B_{4,0}'): 21*Dq, sp.Symbol('F^{(0)}'): 0})
        subs = {
                sp.Symbol('F^{(2)}'): F2,
                sp.Symbol('F^{(4)}'): F4,
                sp.Symbol('\\alpha_T'): 0,
                sp.Symbol('\\zeta_{SO}'): ζ
                }
        matrixmah = matrixmah.subs(subs)
        mfun = sp.lambdify(Dq,matrixmah)
        eigenstates = []
        eigenvalues = []
        eigensystems = []
        for aDq in Dqs:
            nummatrix = mfun(aDq*B)
            nummatrix = np.array(nummatrix, dtype=np.float64)
            eigensys = np.linalg.eigh(nummatrix)
            eigensystems.append(eigensys)
            eigenstates.append(eigensys[1])
            eigenvalues.append(eigensys[0])
        eigenstates = np.array(eigenstates)
        eigenvalues = np.array(eigenvalues)
        
        with h5py.File(eigen_fname,'w') as eigen_h5:
            eigen_h5.create_dataset('/eigenstates',data=eigenstates)
            eigen_h5.create_dataset('/energies',data=eigenvalues)
            eigen_h5.create_dataset('/Dqs_in_cm^-1',data=Dqs*B)
            eigen_h5.create_dataset('/B',data=B)
            eigen_h5.create_dataset('/C',data=C)
            eigen_h5.create_dataset('/zeta',data=ζ)
        progress(total_jobs)
        return eigensystems

if __name__ == '__main__':
    jobs = []
    total_jobs = 0
    for index, row in hf_free_ions.iterrows():
        num_electrons = row['ndN']
        B, C = row['B/cm^-1'], row['C/cm^-1']
        ζ = row['ζd/cm^-1']
        element = row['Element']
        charge = row['Charge']
        jobs.append((B, C, ζ, num_electrons, element, charge))
        if not any(list(map(np.isnan, [B, C, ζ]))):
            total_jobs += 1
    print("Total jobs to run = %d" % total_jobs)
    print("Calculating eigensystems ...")
    all_eigensystems = Parallel(n_jobs=num_cores)(delayed(ion_solver)(*job) for job in jobs)
    print("Finished eigensystem calculation ...")

    print("Creating a single file with all eigenvalues...")

    with h5py.File(energy_h5_file,'w') as h5_file:
        for index, row in hf_free_ions.iterrows():
            ion = Ion(row['Element'], row['Charge'])
            print(row['Element'], row['Charge'])
            eigensys = all_eigensystems[index]
            B, C = row['B/cm^-1'], row['C/cm^-1']
            ionDqs = Dqs*B
            if eigensys:
                eigenvals = np.array([e[0] - np.min(e[0]) for e in eigensys])
                if disambiguate:
                    eigenvals = eigenvalue_disambiguate(np.fliplr(eigenvals))
                energy_address = '/%s/%d/energies' % (ion.symbol, ion.charge_state)
                h5_file.create_dataset(energy_address, data = eigenvals)
                Dqs_address = '/%s/%d/Dqs' % (ion.symbol, ion.charge_state)
                h5_file.create_dataset(Dqs_address, data = ionDqs)
