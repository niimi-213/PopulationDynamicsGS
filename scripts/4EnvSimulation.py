import os
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["VECLIB_MAXIMUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"
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

sys.path.append("..")
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath('__file__')), 'src')
)
from src.PopulationDynamicsModel import *

#3D上で六角形上に点を配置する by GPT-4
def generate_hex_grid_4d(range_limit):
    """
    x + y + z + w = 0 を満たす整数点を生成する関数
    range_limit : 各座標の範囲
    """
    points = {}
    idx = 0
    for x in range(-range_limit, range_limit + 1):
        for y in range(-range_limit, range_limit + 1):
            for z in range(-range_limit, range_limit + 1):
                # wを計算
                w = - (x + y + z)
                # wが範囲内であればポイントを追加
                if -range_limit <= w <= range_limit:
                    points[(x, y, z, w)] = idx
                    idx += 1
    return points
def get_directions_4d():
    """
    4次元空間での方向を返す関数
    """
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
                # x, y, z, w方向の隣接点
    for idx, (x, y, z, w) in enumerate(points):
        directions = get_directions_4d()
        for dx, dy, dz, dw in directions:
            neighbor = (x + dx, y + dy, z + dz,w + dw)
            if neighbor in point_set:
                neighbor_idx = points[neighbor]
                neighbors.append((idx, neighbor_idx))
    return neighbors

hex_grid = generate_hex_grid_4d(20)
print(len(hex_grid))
neighbors = get_neighbors_4d(hex_grid)
print(len(neighbors))

a = 0.01
b = 1
n = len(hex_grid)
E = 4
p = 10**(-5)
A = np.zeros((n,n))
v = np.zeros((n,E))
for i in range(n):
    A[i,i] = 1
for i,j in neighbors:
    A[i,j] = p
    A[i,i] -= p
points = list(hex_grid)
print(A.dtype)
for i in range(n):
    x,y,z,w = points[i]
    v[i] = np.array([2**(x/4),2**(y/4),2**(z/4),2**(w/4)])*(0.5)
gamma = a*np.exp(b*v)
X_0 = np.zeros(n)
X_0[n//2] = 1
S_0 = 1
env_change_num = 20000
tauList = np.full(env_change_num,50)


_,_,N_mean,_ = Simulation(n, E, v, gamma, A, X_0,S_0, tauList, env_change_num, 10**(-8))
    
