import os
import sys
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"

import scipy.optimize as op
import random
from scipy.stats import gmean
from matplotlib import pyplot as plt
import pandas as pd
from scipy.integrate import odeint 
from scipy.integrate import solve_ivp
from scipy.stats import erlang
import numpy as np
import mpmath

sys.path.append("..")
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath('__file__')), 'src')
)
from src.PopulationDynamicsModel import *


#3D上で六角形上に点を配置する by GPT-3
def generate_hexagonal_grid(radius):
    points = []
    for i in range(-radius, radius + 1):
        for j in range(max(-radius, -i-radius), min(radius, -i+radius) + 1):
            k = -i - j
            points.append((i, j, k))
    return points

def get_neighbors(points):
    neighbors = []
    directions = [(1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1)]
    point_set = set(points)
    
    for idx, (x, y, z) in enumerate(points):
        for dx, dy, dz in directions:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in point_set:
                neighbor_idx = points.index(neighbor)
                neighbors.append((idx, neighbor_idx))
    return neighbors

# 格子点の範囲を設定
radius = 18
hex_grid = generate_hexagonal_grid(radius)
neighbors = get_neighbors(hex_grid)
a = 0.01
b = 1
n = len(hex_grid)
E = 3
p = 10**(-4)
A = np.zeros((n,n))
for pair in neighbors:
    i,j = pair
    A[i,j] = p
    A[j,i] = p
for i in range(n):
    A[i,i] = 1 - sum(A[i,:])
v = np.zeros((n,E))
for i in range(n):
    x,y,z = hex_grid[i]
    v[i] = np.array([2**(x/5),2**(y/5),2**(z/5)])*(0.8)
gamma = a*np.exp(b*v)

X_0 = np.zeros(n)
X_0[n//2] = 1
S_0 = 1
env_change_num = 20000
tauList = np.full(env_change_num,50)
    
_,_,N_mean,_ = Simulation(n, E, v, gamma, A, X_0,S_0, tauList, env_change_num, 10**(-8))