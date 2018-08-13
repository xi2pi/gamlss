# -*- coding: utf-8 -*-
"""

@author: Christian Winkler

LMS Method by Cole (1992)

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


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
    return z


# the K matrix only depends on the independent variable x   
def def_K(x):
    h_i = np.ones(len(x))* (x[2]-x[1])
    
    r_ii1 = (1/6)*(h_i)
    r_ii = (2/3)*(h_i)
    r = (np.diag(r_ii) + np.diag(r_ii1, 1)[:-1,:-1] + np.diag(r_ii1, -1)[:-1,:-1])[:-2,:-2]
    
    q_ii1 = 1/h_i
    q_ii = - 2*q_ii1
    q = (np.diag(q_ii1) + np.diag(q_ii, -1)[:-1,:-1] + np.diag(q_ii1, -2)[:-2,:-2])[:,:-2]
    return np.dot(np.dot(q, np.linalg.inv(r)), q.T)   
    

# initialize L0, M0 and S0    
def init_LMS_0(x):
    len_x = len(x)
    L = np.ones(len_x) * 0.9
    M = np.ones(len_x) * 150
    S = np.ones(len_x) * 0.8
    return [L, M, S]
 
# initialize L, M and S       
def init_LMS(x):
    len_x = len(x)
    L = np.ones(len_x) * 1
    M = np.ones(len_x) * 160
    S = np.ones(len_x) * 0.9

    return [L, M, S]
    
def comp_centile(x_LMS, L, M, S, p = 2):
    centile = np.zeros(len(x_LMS))
    for i in range(0, len(x_LMS)):
        if L[i] == 0:
            centile[i] = M[i]*(1+L[i]*S[i]*p)**(1/L[i])
        else:
            centile[i] = M[i]*np.exp(S[i]*p)  
    return centile
  
#compute the loglikelihood function  
def comp_LL(L, M, S, alpha_L, alpha_M, alpha_S, K, y, x, x_LMS):
    z = z_transform(y, x, x_LMS, L, M, S)
    LL_1 = np.zeros(len(x_LMS)) 
    for i in range(0, len(x_LMS)):
        
        # for each i the according data points (x,y) have to be found
        # if there are more than one, they are added up 
        x_ind = np.where(((1.001 * x_LMS[i] >= x) & (x_LMS[i] <= x)))[0]
        LL_1[i] = np.sum(L[i] * np.log(y[x_ind]/M[i])-np.log(S[i])-0.5*z[x_ind]**2)
    
    LL_L_pen = -0.5 * alpha_L * np.dot(np.dot(L.T, K), L)
    LL_M_pen = -0.5 * alpha_M * np.dot(np.dot(M.T, K), M)
    LL_S_pen = -0.5 * alpha_S * np.dot(np.dot(S.T, K), S)
    LL_2 = LL_L_pen + LL_M_pen + LL_S_pen

    
    prob = np.sum(LL_1) + LL_2
    return prob
    
'''
Start
'''

# Data
data = pd.read_csv("example_data.csv")

x = data["age"].values
#y = data["head"].values + np.linspace(50, 150, len(data["head"].values))
y = data["head"].values + 100

# resampling the x vector 
x_LMS = np.linspace(min(x), max(x), len(x))


''' LMS method by Cole 1992 '''

# predefine smoothing parameters alpha
alpha_M = len(x)*((max(x) - min(x))**3 /(400 * (max(y) - min(y))**2)) * 0.2
alpha_S = 2 * alpha_M * np.mean(x) * 100
alpha_L = np.std(x)**4 * alpha_S


K = def_K(x_LMS)

L_0,M_0,S_0 = init_LMS_0(x_LMS)
L,M,S = init_LMS(x_LMS)

# compute loglikelihood function
LL_0 = comp_LL(L, M, S, alpha_L, alpha_M, alpha_S, K, y, x, x_LMS)

diff_LL = 1 # percent


# outer loop
while diff_LL>0.1:
        
    diff_parameter = 100 # percent
    
    # transform y to z
    z = z_transform(y, x, x_LMS, L, M, S)
    
    # compute u and W for equation (9)
    len_x = len(x_LMS)
    u_L = np.zeros(len_x)
    u_M = np.zeros(len_x)
    u_S = np.zeros(len_x)
    W_L = np.zeros(len_x)
    W_M = np.zeros(len_x)
    W_S = np.zeros(len_x)
    W_LM = np.zeros(len_x)
    W_LS = np.zeros(len_x)
    W_MS = np.zeros(len_x)


    for i in range(0, len(x_LMS)):
        
        # for each i the according data points (x,y) have to be found
        # if there are more than one, they are added up 
        x_ind = np.where(((0.01+ x_LMS[i] >= x) & (x_LMS[i] <= x)))[0]
        u_L[i] = np.sum((z[x_ind]/L[i])*(z[x_ind]-np.log(y[x_ind]/M[i])/S[i]) - (np.log(y[x_ind]/M[i])*(z[x_ind]**2 - 1)))
        u_M[i] = np.sum(z[x_ind] / (M[i] * S[i]) + L[i] * (z[x_ind]**2 - 1) / M[i])
        u_S[i] = np.sum((z[x_ind]**2-1)/S[i])
    
        len_x_ind = len(x_ind)
        #len_x_ind = 1
        
        # same for the matrices W 
        if len_x_ind == 0:
            W_L[i] =  (7*(S[i]**2))/4
            W_M[i] = (1 + 2*(L[i]*2)*(S[i]**2))/((M[i]**2)*(S[i]**2))
            W_S[i] =  2/(S[i]**2)
            W_LM[i] =  -1/(2*M[i])
            W_LS[i] =  L[i] * S[i]
            W_MS[i] =  2 * L[i] / (M[i] * S[i])
        else:
            W_L[i] = len_x_ind * (7*(S[i]**2))/4
            W_M[i] = len_x_ind * (1 + 2*(L[i]*2)*(S[i]**2))/M[i]**2/S[i]**2
            W_S[i] = len_x_ind * 2/(S[i]**2)
            W_LM[i] = len_x_ind * -1/(2*M[i])
            W_LS[i] = len_x_ind * L[i] * S[i]
            W_MS[i] = len_x_ind * 2 * L[i] / (M[i] * S[i])

        
    W_L = np.diag(W_L)
    W_M = np.diag(W_M)
    W_S = np.diag(W_S)
    W_LM = np.diag(W_LM)
    W_LS = np.diag(W_LS)
    W_MS = np.diag(W_MS)
    W_ML = W_LM
    W_SL = W_LS
    W_SM = W_MS

    
    # inner loop
    while diff_parameter>10:
        
        ''' equation (9) in Cole (1992)'''
        
        G_L = np.linalg.inv(W_L+alpha_L*K)
        G_M = np.linalg.inv(W_M+alpha_M*K)
        G_S = np.linalg.inv(W_S+alpha_S*K)
    
        L_update = np.dot(G_L,(u_L+np.dot(W_L,L_0)-np.dot(W_LM, (M-M_0))-np.dot(W_LS, (S-S_0))))
        M_update = np.dot(G_M,(u_M+np.dot(W_M,M_0)-np.dot(W_MS, (S-S_0))-np.dot(W_ML, (L-L_0))))
        S_update = np.dot(G_S,(u_S+np.dot(W_S,S_0)-np.dot(W_SL, (L-L_0))-np.dot(W_SM, (M-M_0))))
                
        # I think, here is a mistake. For some reason it does not converge. Maybe I misunderstood this step
                
        # plotting the updated values L
        plt.plot(x_LMS,M, 'r', label = "M old")
        plt.plot(x_LMS,M_update, label = "M")
        plt.plot(x,y, '.')
        
        plt.legend()
        plt.show()
        
        plt.plot(x_LMS, S, label = "S old")
        plt.plot(x_LMS, S_update, label = "S")
        plt.plot(x_LMS, L, label = "L old")
        plt.plot(x_LMS, L_update, label = "L")
        
        plt.legend()
        plt.show()
        
        
        diff_parameter = np.sum(np.abs((M-M_update)/M_update))
            
        L = L_update 
        M = M_update
        S = S_update

#        print(min(L))
#        print(min(M))
#        print(min(S))
        
        print(diff_parameter)
        print("inner loop - next iteration step")
    
    # problem: inner loop does not converge
    # diff_LL = 0.001
    print("inner loop - done")
    LL_calc = comp_LL(L, M, S, alpha_L, alpha_M, alpha_S, K, y, x, x_LMS)
    diff_LL = np.abs((LL_0-LL_calc)/LL_calc)
    
    L_0 = L
    M_0 = M
    S_0 = S
    
    LL_0 = LL_calc
    print("LogLikelihood: " + str(diff_LL))
    

plt.plot(x_LMS,M, label = "M")
plt.plot(x_LMS,comp_centile(x_LMS, L, M, S, p = 2), label = "z-score: " + str(2))
plt.plot(x_LMS,comp_centile(x_LMS, L, M, S, p = 1), label = "z-score: " + str(1))
plt.plot(x_LMS,comp_centile(x_LMS, L, M, S, p = 0.5), label = "z-score: " + str(0.5))
plt.plot(x_LMS,comp_centile(x_LMS, L, M, S, p = -0.5), label = "z-score: " + str(-0.5))
plt.plot(x_LMS,comp_centile(x_LMS, L, M, S, p = -1), label = "z-score: " + str(-1))
plt.plot(x_LMS,comp_centile(x_LMS, L, M, S, p = -2), label = "z-score: " + str(-2))
plt.plot(x,y, '.')

plt.legend()
plt.show()

plt.plot(x_LMS, S_update, label = "S")
plt.legend()
plt.show()

plt.plot(x_LMS, L_update, label = "L")
plt.legend()
plt.show()


