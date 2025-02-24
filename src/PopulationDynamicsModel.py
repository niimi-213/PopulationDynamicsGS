import numpy as np
import random
import scipy.optimize as op


# Calculate S(t)
def S_cons(t,S0,v,U,P0,L):
    int_N = U @ ((P0*(np.exp(L*t)-1))/L)
    return S0 - np.dot(v,int_N)
def interval(tau,S0,v,U,P0,L):
    t = 0
    dt = tau/20
    while t <= tau:
        t += dt
        if S_cons(t,S0,v,U,P0,L) < 0:
            return t
    return tau
# Calculate the average population
def N_average(U,P0,L,t_p,gamma,tau):
    N_p = U @ (P0*(np.exp(L*t_p)-1)/L)
    N_m = (U @ (P0*np.exp(L*t_p))) * (1-np.exp(-gamma*(tau-t_p)))/ gamma
    return (N_p + N_m) / tau
# Simulation
def Simulation(n, E, v, gamma, A, N_0, S0, tauList, env_change_num,threshold = 10**(-8)):
    """
    - n: 表現型数
    - v: 成長速度(n*Eのサイズの行列)
    - mu: 死亡速度(n*Eのサイズの行列)
    - A: 遷移行列
    - E: 環境数
    - N0: 初期個体数(サイズnのベクトル)
    - S0: 栄養量(環境切り替わりごとにS_jの値にセットされる
    - tauList 環境持続時間(List: 1,env_change_num)
    - env_change_num 環境が切り替わる回数
    - threshold 個体数を0にする閾値
    """
    N_course = [N_0]
    t_course = [0]
    N = N_0 #initial population
    N_mean = []
    tau_course = []
    total_time = 0

    A_v_list = []

    for i in range(E):
        Av = (A.T)@np.diag(v[:,i])
        eig = np.linalg.eig(Av)
        #行列　対角化 Av = U @ diag(L) @ U^{-1}
        L = eig[0]
        U = eig[1]
        A_v_list.append((Av,L,U))
    env = 0
    for i in range(env_change_num):
        env = random.randint(0, E-1)
        #env = i%E
        tau = tauList[i]
        Av,L,U = A_v_list[env]  
        P0 = np.linalg.solve(U,N)
            
        #栄養取り尽くしt_p
        tt = interval(tau,S0,v[:,env],U,P0,L)
        if S_cons(tt,S0,v[:,env],U,P0,L) <= 0:
            t_p = op.brentq(S_cons,0,tt,args=(S0,v[:,env],U,P0,L))
        else:
            t_p = tau
        tau_course.append(tau)
        #成長
        N = U @ (P0*np.exp(L*t_p)) #N(t_p)
        N = np.where(N<threshold,0,N)
        N_course.append(N)
        t_course.append(t_p + total_time)
        #死亡
        N = N * np.exp(-gamma[:,env]*(tau-t_p))
        N = np.where(N<threshold,0,N)
        N_course.append(N)
        t_course.append(tau + total_time)
        total_time += tau
        N = np.where(N<threshold,0,N)
              
        if np.all(N<threshold):
            print("extinct")
            for _ in range(env_change_num-i):
                N_course.append(np.zeros(n))
                N_mean.append(np.zeros(n))
                t_course.append(tau + total_time)
                total_time += tau
            break
        N_mean.append(N_average(U,P0,L,t_p,gamma[:,env],tau))

    N_course = np.array(N_course)
    t_course = np.array(t_course)
    N_mean = np.array(N_mean)
    return N_course,t_course,N_mean,tau_course