import os
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
import tqdm
import time
from scipy.integrate import odeint 
from scipy.integrate import solve_ivp
from scipy.stats import erlang
import numpy as np
import mpmath
import sys
import copy

sys.path.append("..")
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath('__file__')), 'src')
)
from src.PopulationDynamicsModel import *

def generate_hex_grid_4d(range_limit):
    """
    generate lattice point (x,y,z,w), x + y + z + w = 0
    range_limit : range of each axis
    """
    points = {}
    idx = 0
    for x in range(-range_limit, range_limit + 1):
        for y in range(-range_limit, range_limit + 1):
            for z in range(-range_limit, range_limit + 1):
                w = - (x + y + z)
                if -range_limit <= w <= range_limit:
                    points[(x, y, z, w)] = idx
                    idx += 1
    return points
def get_directions_4d():
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    if (dx != 0 or dy != 0 or dz != 0 or dw != 0) and (dx+dy+dz+dw) == 0 and (abs(dx) + abs(dy) + abs(dz) + abs(dw) == 2):
                        directions.append((dx, dy, dz, dw))
    return directions
def get_neighbors_4d(points):
    point_set = set(points)
    neighbors = []
    for idx, (x, y, z, w) in enumerate(points):
        directions = get_directions_4d()
        for dx, dy, dz, dw in directions:
            neighbor = (x + dx, y + dy, z + dz,w + dw)
            if neighbor in point_set:
                neighbor_idx = points[neighbor]
                neighbors.append((idx, neighbor_idx))
    return neighbors
def adj_k(n,k,neighbors):
    """ Returns a list of k-nearest neighbors
    n: number of points
    k: k-nearest neighbors
    neighbors: list of neighbors
    """
    adjL = [[i] for i in range(n)]
    for pair in neighbors:
        x,y = pair
        adjL[x].append(y)
    if k == 1:
        return adjL
    next = copy.deepcopy(adjL)
    for i in range(k-1):
        tmp = [[] for i in range(n)]
        for j in range(n):
            for x in adjL[j]:
                for y in next[x]:
                    tmp[j].append(y)
            tmp[j] = list(set(tmp[j]))
        adjL = copy.deepcopy(tmp)
    return adjL

hex_grid = generate_hex_grid_4d(18)
neighbors = get_neighbors_4d(hex_grid)
points = list(hex_grid)

a = 0.01
b = 1
n = len(points)
E = 4
p = 10**(-5)
A = np.zeros((n,n))
for pair in neighbors:
    x,y = pair
    A[x,y] = p
    A[y,x] = p
for i in range(n):
    A[i,i] = 1 - sum(A[i,:])
mu = np.zeros((n,E))

for i in range(n):
    x,y,z,w = points[i]
    mu[i] = np.array([2**(x/3),2**(y/3),2**(z/3),2**(w/3)])*(0.2)
gamma = a*np.exp(b*mu)
X_0 = np.zeros(n)
X_0[n//2] = 1

env_change_num = 20000
tauList = np.full(10,env_change_num)
_,_,N_mean,_ = Simulation_approx(n, E, mu, gamma, A, X_0, 1, tauList, env_change_num,adj_k(n,1,neighbors),threshold=10**(-8))
    
