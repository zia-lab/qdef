#!/usr/bin/env python3

from qdef import *
import sympy as sp
import pandas as pd
from matrixgoodies import *
from misc import *
from joblib import Parallel, delayed
import multiprocessing
from time import time
import h5py

info = '''┌────────────────────────────────────────────────────────┐
│                                                        │
│  This script calculates Tanabe-Sugano diagrams for a   │
│    crystal field with cubic symmetry for a range of    │
│             values for C/B, Dq/B, and ζ/B.             │
│                                                        │
│   For each combination an .h5 file is produced with    │
│  the eigenvalues and eigenstates. The keys needed to   │
│   access these should be apparent from inspection of   │
│                      this script.                      │
│                                                        │
│      At then end all eigenvalues (discarding the       │
│      eigenstates) are saved to a large .h5 file.       │
│                                                        │
│   All calculations are done in the uncoupled basis.    │
│                                                        │
│   This was run at CCV, and takes about six hours to    │
│                        finish.                         │
│                                                        │
│                         David.                         │
│                                                        │
│               Mar-05 2022-03-05 20:48:06               │
│                                                        │
└────────────────────────────────────────────────────────┘'''

mesh_count = 111
Dqs_B = list(np.linspace(-6,6,2*mesh_count+1))
ζs_B = list(np.linspace(0,11,mesh_count))
γs_B = list(np.linspace(3.41,4.51,mesh_count))
num_electrons_s = [1,2,3,4,5,6,7,8,9]
num_precision = np.float32 # only applies to final export of eigenvalues
out_folder = '/users/jlizaraz/scratch/out/qdef/tsk/'
h5_filename = '/users/jlizaraz/scratch/out/qdef/tsk_hypercube.h5'

import http.client, urllib
import sys

def send_message(message):
    conn = http.client.HTTPSConnection("api.pushover.net",443)
    conn.request("POST", "/1/messages.json",
      urllib.parse.urlencode({
        "token": "aqxvnvfq42adpf78g9pwmphse9c2un",
        "user": "uqhx6qfvn87dtfz5dhk71hf2xh1iwu",
        "message": message,
      }), { "Content-type": "application/x-www-form-urlencoded" })
    conn.getresponse()
    return None


if __name__ == '__main__':
    print(info)
    num_cores = multiprocessing.cpu_count()
    Dq = sp.Symbol('Dq')
    l = 2

    def cubic_solver(num_electrons, F2, F4, ζ, Dqs, idx_γ, idx_ratio):
        job_id = '%d-%d-%d' % (num_electrons, idx_γ, idx_ratio)
        milestone_file = '%s.h5' % job_id
        milestone_file = os.path.join(out_folder, milestone_file)
        params_array = np.array([γs_B[idx_γ], ζs_B[idx_ratio]])
        if any(list(map(np.isnan, [F2, F4, ζ]))):
            open(milestone_file,'w').write(str(time()))
            return None
        else:
            ham = hamiltonian_CF_CR_SO_TO(num_electrons, 'O', l, False, True)
            matrixmah = ham[0].subs({sp.Symbol('B_{4,0}'): 10*Dq, sp.Symbol('F^{(0)}'): 0})
            subs = {
                    sp.Symbol('F^{(2)}'): F2,
                    sp.Symbol('F^{(4)}'): F4,
                    sp.Symbol('\\alpha_T'): 0,
                    sp.Symbol('\\zeta_{SO}'): ζ
                    }
            matrixmah = matrixmah.subs(subs)
            mfun = sp.lambdify(Dq,matrixmah)
            all_eigenvalues = []
            all_eigenstates = []
            B = F2/49. - 5*F4/441.
            for aDq in Dqs:
                nummatrix = mfun(aDq)
                eigensys = np.linalg.eigh(nummatrix)
                eigenvals = eigensys[0]
                eigenstates = eigensys[1]
                all_eigenvalues.append(eigenvals)
                all_eigenstates.append(eigenstates)
            all_eigenstates = np.array(all_eigenstates)
            all_eigenvalues = np.array(all_eigenvalues)
            with h5py.File(milestone_file,'w') as h5file:
                h5file.create_dataset('/eigenvalues', data = all_eigenvalues)
                h5file.create_dataset('/eigenstates', data = all_eigenstates)
                h5file.create_dataset('/Dqs', data = Dqs)
                h5file.create_dataset('/gamma_and_CoB', data = params_array)
            h5file.close()
            return ((num_electrons, idx_γ, idx_ratio), all_eigenvalues)

    jobs = []
    for γ_B, ζ_B, num_electrons in product(γs_B, ζs_B, num_electrons_s):
        idx_ratio = ζs_B.index(ζ_B)
        idx_γ = γs_B.index(γ_B)
        F2 = 49 + 7*γ_B
        F4 = 63*γ_B/5
        ζ = ζ_B
        job_id = '%d-%d-%d' % (num_electrons, idx_γ, idx_ratio)
        milestone_file = '%s.h5' % job_id
        milestone_file = os.path.join(out_folder, milestone_file)
        if os.path.exists(milestone_file):
            continue
        jobs.append((num_electrons, F2, F4, ζ, Dqs_B, idx_γ, idx_ratio))
    
    print("Running %d jobs in %d cores." % (len(jobs), num_cores))

    start_time = time()
    all_all_eigenvalues = Parallel(n_jobs = num_cores)(delayed(cubic_solver)(*job) for job in jobs)
    end_time = time()
    
    time_taken = end_time - start_time
    print("time taken %.1f min" % (time_taken/60))
    print("Finished eigensystem calculation!")  
    
    print("Saving to all_eigenvalues to %s" % h5_filename)
    with h5py.File(h5_filename,'w') as h5_file:
        for key, eigvees in all_all_eigenvalues:
            num_electrons, idx_γ, idx_ratio = key
            address = '/%d/%d/%d' % key
            h5_file.create_dataset(address, data = eigvees)
        h5_file.create_dataset('/params/Dqs_B', data = Dqs_B) 
        h5_file.create_dataset('/params/zetas_B', data = ζs_B) 
        h5_file.create_dataset('/params/gammas_B', data = γs_B)  
        h5_file.close()
    send_message('finished LONG march')

