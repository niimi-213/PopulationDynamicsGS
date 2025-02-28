import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import scipy.optimize as op
import random
from matplotlib import pyplot as plt
import tqdm
import numpy as np
import sys
sys.path.append("..")
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath('__file__')), 'src')
)
from src.PopulationDynamicsModel import *


trial_num = 40
n = 51
E = 2
Ns = np.zeros((trial_num,n))
vs1 = np.zeros((trial_num,n))
vs2 = np.zeros((trial_num,n))
v_m = np.linspace(0.6,0.8,trial_num)
v_gmean = np.logspace(-1.5,0.4,trial_num)
for t in tqdm.tqdm(range(trial_num)):
    random.seed(100)
    np.random.seed(100)
    A = np.zeros((n,n))
    p = 10**(-4)
    for i in range(n):
        for j in range(n):
            if abs(i-j)==1:
                    A[i,j] = p
            else:
                A[i,j] = 0

    for i in range(n):
        A[i,i] = 1 - sum(A[i,:])
        
    a = 0.01
    b=1
    v = np.zeros((n,E))
    gamma = np.zeros((n,E))

    for i in range(n):
        p = (i+1)*v_m[t]/(n+1)
        v[i] = np.array([(2)**(p/4),(2)**(-p/4)])*v_gmean[t]

    gamma = a*np.exp(b*v)
    X_0 = np.zeros(n)
    for i in range(n):
        X_0[i] = 1
    env_change_num=40000
    tauList = np.full(env_change_num,10)   
    N,_,N_mean,tau_course = Simulation(n, E, v, gamma, A, X_0, 1, tauList, env_change_num,threshold=10**(-8))
    Ns[t] = N_mean[-1,:]/2 + N_mean[-2,:]/2
    vs1[t] = v[:,0]
    vs2[t] = v[:,1]