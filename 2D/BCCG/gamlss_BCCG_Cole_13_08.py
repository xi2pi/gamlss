# -*- coding: utf-8 -*-
"""

@author: Christian Winkler

LMS Method by Cole (1992)

"""
import matplotlib.pyplot as plt
from scipy import exp
from scipy.special import gamma, erf
import numpy as np
import pandas as pd
#from scipy.optimize import curve_fit
import scipy.stats as stats
import scipy.optimize as optimization
from scipy import interpolate
from scipy import stats
#import time


    
def z_transform(y, x, x_LMS, L, M, S):
    z = np.zeros(len(x_LMS))
    f_L = interpolate.interp1d(x_LMS, L,  kind = "linear")
    f_M = interpolate.interp1d(x_LMS, M,  kind = "linear")
    f_S = interpolate.interp1d(x_LMS, S,  kind = "linear")
    
    L = f_L(x)
    M = f_M(x)
    S = f_S(x) 
    for i in range(0, len(x_LMS)):
        if L[i] == 0:
            z[i] = (1/S[i])*np.log(y[i]/M[i])
        else:
            z[i] = (1/(S[i]*L[i]))*(((y[i]/M[i])**L[i])-1)

#    plt.plot(x, L, '.')
#    plt.show()

    return z


    
def def_K(x):
    #h_i = np.diff(x)
    h_i = np.ones(len(x))* (x[2]-x[1])

    #h_i1 = np.roll(h_i, -1)
    h_i1 = h_i

    
    r_ii1 = (1/6)*(h_i)
    r_ii = (2/3)*(h_i)
    r = (np.diag(r_ii) + np.diag(r_ii1, 1)[:-1,:-1] + np.diag(r_ii1, -1)[:-1,:-1])[:-2,:-2]
    
    q_ii1 = 1/h_i
    q_ii = - 2*q_ii1
    q = (np.diag(q_ii1) + np.diag(q_ii, -1)[:-1,:-1] + np.diag(q_ii1, -2)[:-2,:-2])[:,:-2]
    return np.dot(np.dot(q, np.linalg.inv(r)), q.T)
#    
    
def x_resample(x):
    return np.linspace(min(x), max(x), len(x))
    #return np.linspace(min(x), max(x), 10)
    
def init_LMS_0(x):
    len_x = len(x)
    #print(len_x)
    L = np.ones(len_x) * 0.001
    M = np.ones(len_x) * 150
    S = np.ones(len_x) * 0.00001
#    L = np.ones(len_x) * 1
#    M = np.ones(len_x) * 150
#    S = np.ones(len_x) * 0.2
    
    return [L, M, S]
       
def init_LMS(x):
    len_x = len(x)
    #print(len_x)
    L = np.ones(len_x) * 0.0001
    M = np.ones(len_x) * 160
    S = np.ones(len_x) * 0.00002
#    L = np.ones(len_x) * 1
#    M = np.ones(len_x) * 150
#    S = np.ones(len_x) * 0.2
     
    return [L, M, S]
    
def comp_LL(L, M, S, alpha_L, alpha_M, alpha_S, K, y, x, x_LMS):
    #LL_1 = np.zeros(len(x))
    z = z_transform(y, x, x_LMS, L, M, S)
    
    LL_1 = L * np.log(y/M)-np.log(S)-0.5*z**2
    
    LL_L_pen = -0.5 * alpha_L * np.dot(np.dot(L.T, K), L)
    LL_M_pen = -0.5 * alpha_M * np.dot(np.dot(M.T, K), M)
    LL_S_pen = -0.5 * alpha_S * np.dot(np.dot(S.T, K), S)
    LL_2 = LL_L_pen + LL_M_pen + LL_S_pen

    #print(LL_2)
    
    prob = np.sum(LL_1) + LL_2
    return prob
    

# Test
data = pd.read_csv("example_data.csv")

x = data["age"].values
y = data["head"].values + 100

#y = [x for _,x in sorted(zip(x,y))]



x_LMS = x_resample(x)
#y = y_transform(y,x, x_LMS)

'''
Start
'''

# Siehe Pub Cole 1992

alpha_M = len(x)*(max(x) - min(x))**3 /(400 * (max(y) - min(y))**2)

alpha_S = 2 * alpha_M * np.mean(x)
alpha_L = np.std(x)**4 * alpha_S
#alpha_S = alpha_M
#alpha_L = alpha_M
#alpha_L = 0.01


K = def_K(x_LMS)

L_0,M_0,S_0 = init_LMS_0(x_LMS)
L,M,S = init_LMS(x_LMS)

#f_L = interpolate.interp1d(x_LMS, L,  kind = "linear")
#f_M = interpolate.interp1d(x_LMS, M,  kind = "linear")
#f_S = interpolate.interp1d(x_LMS, S,  kind = "linear")
#
#L_trans = f_L(x)
#M_trans = f_M(x)
#S_trans = f_S(x) 

LL_0 = comp_LL(L, M, S, alpha_L, alpha_M, alpha_S, K, y, x, x_LMS)

diff_LL = 1 # percent


# outer loop
while diff_LL>0.04:
        
    diff_parameter = 1 # percent
    
    z = z_transform(y, x, x_LMS, L, M, S)
    
    z2 = np.array([x for _,x in sorted(zip(x,z))])
    values = z2
    z2 = stats.binned_statistic(x, values, 'median', bins=np.linspace(min(x), max(x), len(x)+1))[0]
    
    plt.plot(x, z2)
    plt.show()
    
    u_L = (z/L)*(z-np.log(y/M)/S) - (np.log(y/M)*(z**2 - 1))
    u_M = z / (M * S) + L * (z**2 - 1) / M
    u_S = (z**2-1)/S
    W_L = np.diag((7*(S**2))/4)
    W_M = np.diag((1 + 2*(L**2)*(S**2))/M**2/S**2)
    W_S = np.diag(2/S**2)
    W_LM = np.diag(-1/(2*M))
    W_LS = np.diag(L * S)
    W_MS = np.diag(2 * L / (M * S))
    W_ML = W_LM
    W_SL = W_LS
    W_SM = W_MS
    
#    u_L = (z/L)*(z-np.log(y/M)/S) - (np.log(y/M)*(z**2 - 1))
#    u_M = z / (M * S) + L * (z**2 - 1) / M
#    u_S = (z**2-1)/S
#    W_L = np.diag((7*(S**2))/4)
#    W_M = np.diag((1 + 2*(L**2)*(S**2))/M**2/S**2)
#    W_S = np.diag(2/S**2)
#    W_LM = np.diag(-1/(2*M))
#    W_LS = np.diag(L * S)
#    W_MS = np.diag(2 * L / (M * S))
#    W_ML = W_LM
#    W_SL = W_LS
#    W_SM = W_MS
    
    # inner loop
    # fehlerhaft
    while diff_parameter>0.5:
        
        G_L = np.linalg.inv(W_L+alpha_L*K)
        G_M = np.linalg.inv(W_M+alpha_M*K)
        G_S = np.linalg.inv(W_S+alpha_S*K)

        L_calc = np.dot(G_L,(u_L+np.dot(W_L,L)-np.dot(W_LM, (M-M_0))-np.dot(W_LS, (S-S_0))))
        M_calc = np.dot(G_M,(u_M+np.dot(W_M,M)-np.dot(W_MS, (S-S_0))-np.dot(W_ML, (L-L_0))))
        S_calc = np.dot(G_S,(u_S+np.dot(W_S,S)-np.dot(W_SL, (L-L_0))-np.dot(W_SM, (M-M_0))))

        plt.plot(M, 'r')
        plt.plot(M_calc)
        plt.plot(y, '.')
        
        plt.show()
        
        plt.plot(S, label = "S old")
        plt.plot(S_calc, label = "S")
        plt.plot(L, label = "L old")
        plt.plot(L_calc, label = "L")
        
        plt.legend()
        plt.show()
        
        print(min(L_calc))
        print(min(M_calc))
        
        #diff_parameter = max([np.sum(np.abs((L-L_calc)/L_calc))/len(L_calc), np.sum(np.abs((M-M_calc)/M_calc))/len(M_calc), np.sum(np.abs((S-S_calc)/S_calc))/len(S_calc)])
        diff_parameter = np.sum(np.abs((M-M_calc)/M_calc))


        L = L_calc 
        M = M_calc
        S = S_calc
        
        
        print(diff_parameter)
        print("next")
        
    diff_LL = 0.001
        
#    M = M - min(M) + 1
#    S = np.linspace(0.1, 1, len(M))
#    L = np.linspace(0.1, 1, len(M))
#    # hier klappt es nicht, weil M negative Werte aufweist
#    LL_calc = comp_LL(L, M, S, alpha_L, alpha_M, alpha_S, K, y, x, x_LMS)
#    diff_LL = np.abs((LL_0-LL_calc)/LL_calc)
#    
#    LL_0 = LL_calc
#    print("LogLikelihood: " + str(diff_LL))

