#!/usr/bin/env python3

import numpy as np
import sympy as sp
import time
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
import random

num_cores = multiprocessing.cpu_count()

def fft():
    for _ in range(10):
        ft = np.fft.fft(np.random.random(100000))
fft.desc = 'Calculating the fast Fourier transform of a big array ... '

def eig():
    for _ in range(10):
        eigvs = np.linalg.eigvals(np.random.random((100,100)))
eig.desc = 'Calculating the eigenvalues of a 100x100 square matrix ... '

def rando():
    randos = np.random.random((2000,2000))
rando.desc = 'Calculating a 2000x2000 random array ... '

def multi():
    a0 = np.random.random((100,100))
    a1 = np.random.random((100,100))
    for _ in range(10000):
        b = a0 * a1
multi.desc = 'Multiplying arrays with lots of repeats ... '

def matinv():
    m0 = np.random.random((100,100))
    for i in range(10):
        np.linalg.inv(m0)
matinv.desc = 'Calculating the matrix inverse of a rando matrix ... '

def sorter():
    ar = np.random.random(10000)
    for _ in range(100):
        np.sort(ar)
sorter.desc = 'Sorting a big random array ... '

def itersum():
    it = range(100)
    summa = 0
    for i0, i1, i2 in product(it, it, it):
        summa += i0+i1+i2
itersum.desc = 'Summing lots of numbers in 3 nested loops ... '

def funceval():
    funcs = [np.sin, np.cos, np.tan, lambda x: 1/x, np.log10, np.log2]
    theta = np.linspace(0.1,2*np.pi,1000000)
    for func in funcs:
        func(theta)
funceval.desc = 'Calculating function evaluation many times over ... '

def symbexpand():
    monos = [np.random.randint(0,10)*sp.Symbol('x')+np.random.randint(1,10) for _ in range(20)]
    poly = sp.S(1)
    for mono in monos:
        poly = poly*mono
    sp.expand(poly)
symbexpand.desc = 'Expanding a convoluted symbolic expression ... '

def picalc():
    longpi = sp.N(sp.sqrt(sp.pi), 100000)
picalc.desc = 'Calcuating one hundred thousand digits of sqrt(pi) ... '

benchmarks = {'fft': fft, 'eig': eig, 'rando': rando,
              'multi': multi, 'matinv': matinv, 'sorter': sorter,
              'itersum': itersum, 'funceval': funceval,
              'symbexpand': symbexpand, 'picalc': picalc}

def npbenchmark():
    '''
    Standard score is 1000, which is performance of 2021 MacBook Pro 16".
    '''
    repeats = 10
    BASE_TIME = 5.014
    print("#### NumPy benchmark v 0.1 ####")
    print('-'*31)
    print("> Testing serial execution:")
    print('-'*31)
    timings = {}
    for bench, benchfun in benchmarks.items():
        # print('%s  ...  ' % bench, end='')
        print('>> ' + benchfun.desc, end='')
        times = []
        for _ in range(repeats):
            start_time = time.process_time()
            benchfun()
            elapsed_time = time.process_time() - start_time
            times.append(elapsed_time)
        elapsed_time = np.sum(times)
        msg = '%.3f s' % elapsed_time
        print(msg)
        timings[bench] = elapsed_time
    timings['total'] = sum(timings.values())
    timings['score'] = np.round(1000 * BASE_TIME / timings['total'])
    print('-'*31)
    print('TOTAL TIME  = %.3f s' % timings['total'])
    print('SERIAL SCORE = %d' % timings['score'])
    print('-'*31)
    print("> Testing execution in parallel:")
    print("> Using %d cores." % num_cores)
    print('-'*31)
    def indexed_benchmark(key):
        benchmarks[key]()
    jobs = sum([list(benchmarks.keys()) for _ in range(repeats)],[])
    random.shuffle(jobs)
    start_time = time.process_time()
    Parallel(n_jobs = num_cores)(delayed(indexed_benchmark)(job) for job in jobs)
    parallel_elapsed_time = time.process_time() - start_time
    timings['parallel_elapsed_time'] = parallel_elapsed_time
    timings['parallel_score'] = 1000 * BASE_TIME / parallel_elapsed_time
    print('TOTAL TIME  = %.3f s' % parallel_elapsed_time)
    print('PARALLEL SCORE = %d' % timings['parallel_score'])
    print('-'*31)
    return timings

npbenchmark()
