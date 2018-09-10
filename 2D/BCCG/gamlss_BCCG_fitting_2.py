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
#import time

#def NO(x,M,S):
#    return 1/(np.sqrt(2*np.pi)*S)*exp(-(x-M)**2/(2*S**2))
#    
#def BCCG(params,x):
#    L = params[0]
#    M = params[1]
#    S = params[2]
#    
#    Phi = 0.5*(1 + erf((1/(S*np.abs(L)))/(np.sqrt(2))))
#    
#    if L == 0:
#        z = (1/S)*np.log(x/M)
#    else:
#        z = (1/(S*L))*(((x/M)**L)-1)
#    
#    f = (x**(L-1)*np.exp(-0.5*z**2))/((M**L)*S*Phi*np.sqrt(2*np.pi))
#    return f
    
def z_transform(y, x, x_LMS, L, M, S):
    z = np.zeros(len(L))
    idx = [(np.abs(x[i] - x_LMS)).argmin() for i in range(0, len(y))]
   
    for i in range(0, len(L)):
        if L[idx[i]] == 0:
            z[i] = (1/S[idx[i]])*np.log(y[i]/M[idx[i]])
        else:
            z[i] = (1/(S[idx[i]]*L[idx[i]]))*(((y[i]/M[idx[i]])**L[idx[i]])-1)
    return z
    
#def z_transform(y, x, x_LMS, L, M, S):
#    z = np.zeros(len(L))
#    #idx = [(np.abs(x[i] - x_LMS)).argmin() for i in range(0, len(y))]
#   
#    for i in range(0, len(L)):
#        if L[i] == 0:
#            z[i] = (1/S[i])*np.log(y[i]/M[i])
#        else:
#            z[i] = (1/(S[i]*L[i]))*(((y[i]/M[i])**L[i])-1)
#    return z
#    
#def def_N(x):
#    #h_i = np.ones(len(x))* (x[2]-x[1])
#    h_i = np.ones(len(x))
#    q = np.diag(h_i)
##    h_i1 = h_i
##    
##    q_ii1 = 1/h_i
##    q_ii = - 2*q_ii1
##    q = (np.diag(q_ii) + np.diag(q_ii1, 1)[:-1,:-1] + np.diag(q_ii1, -1)[:-1,:-1])
##    q[0,0]=0
##    q[0,1]=0
##    q[1,0]=0
##    q[-1,0]=0
##    q[0,-1]=0
##    q[-1,-1]=0
##    q[-2,-2]=0
##    q[2,2]=0
##    q[0,2]=0
##    q[2,0]=0
##    q[-2,0]=0
##    q[0,-2]=0
    
#
#    
#    return q
    
    
def def_K(x):
    #h_i = np.diff(x)
    h_i = np.ones(len(x))* (x[2]-x[1])

    #h_i1 = np.roll(h_i, -1)
    h_i1 = h_i
    
    # check dimension of matrices (wikipedia smoothin splines)    
#    delta_ii = 1/h_i
#    delta_ii2 = 1/h_i1
#    delta_ii1 = -delta_ii - delta_ii2
#
#    
#    delta = (np.diag(delta_ii) + np.diag(delta_ii1, 1)[:-1,:-1] + np.diag(delta_ii2, 2)[:-2,:-2])[:-2]
#    #print(delta)
#    
#    W_ii = (h_i + h_i1)/3
#    W_i1i = h_i/6
#    
#    W = (np.diag(W_ii) + np.diag(W_i1i,1)[:-1,:-1] + np.diag(W_i1i,-1)[:-1,:-1])[:-2,:-2]
#    #print(W)
#    return np.dot(np.dot(delta.T, np.linalg.inv(W)), delta)
    
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
    
def init_LMS_0(x):
    len_x = len(x)
    print(len_x)
    L = np.zeros(len_x)
    M = np.zeros(len_x) 
    S = np.zeros(len_x) 
    
#    L_vect = np.dot(N, L)
#    M_vect = np.dot(N, M)
#    S_vect = np.dot(N, S)
    # I think it depends on the starting values for LMS
    return [L, M, S]
       
def init_LMS(x):
    len_x = len(x)
    print(len_x)
    L = np.ones(len_x) * 1
    M = np.ones(len_x) * 150
    S = np.ones(len_x) * 10
    
#    L_vect = np.dot(N, L)
#    M_vect = np.dot(N, M)
#    S_vect = np.dot(N, S)
    # I think it depends on the starting values for LMS
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
    
#def err_func(x_opt, L_0, M_0, S_0, y, x):
#    print("call error function")
#    
#    array_split = np.split(x_opt, 3)
#    L = array_split[0]
#    M = array_split[1]
#    S = array_split[2]
#    
#    z = z_transform(y, x, x_LMS, L, M, S)
#    
#    u_L = (z/L)*(z-np.log(y/M)/S)- np.log(y/M)*(z**2 - 1)
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
#    
#    err_L = L - np.dot(np.linalg.inv(W_L+alpha_L*K),(u_L+np.dot(W_L,L)-np.dot(W_LM, (M-M_0))-np.dot(W_LS, (S-S_0))))
#    err_M = M - np.dot(np.linalg.inv(W_M+alpha_M*K),(u_M+np.dot(W_M,M)-np.dot(W_MS, (S-S_0))-np.dot(W_ML, (L-L_0))))
#    err_S = S - np.dot(np.linalg.inv(W_S+alpha_S*K),(u_S+np.dot(W_S,S)-np.dot(W_SL, (L-L_0))-np.dot(W_SM, (M-M_0))))
#    print(np.sum(err_L**2)+np.sum(err_M**2)+np.sum(err_S**2))
#    return np.sum(err_L**2)+np.sum(err_M**2)+np.sum(err_S**2)
    
def err_func(para, L_0, M_0, S_0, y, x):
    print("call error function")
    
    a = para[0]
    b = para[1]
    c = para[2]
    d = para[3]
    e = para[4]
    f = para[5]
    
    #array_split = np.split(x_opt, 3)
    L = a* np.linspace(0, 1, len(x)) + b
    M = c* np.linspace(0, 1, len(x)) + d
    S = e* np.linspace(0, 1, len(x)) + f
    
    z = z_transform(y, x, x_LMS, L, M, S)
    
    u_L = (z/L)*(z-np.log(y/M)/S)- np.log(y/M)*(z**2 - 1)
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
    
    err_L = L - np.dot(np.linalg.inv(W_L+alpha_L*K),(u_L+np.dot(W_L,L)-np.dot(W_LM, (M-M_0))-np.dot(W_LS, (S-S_0))))
    err_M = M - np.dot(np.linalg.inv(W_M+alpha_M*K),(u_M+np.dot(W_M,M)-np.dot(W_MS, (S-S_0))-np.dot(W_ML, (L-L_0))))
    err_S = S - np.dot(np.linalg.inv(W_S+alpha_S*K),(u_S+np.dot(W_S,S)-np.dot(W_SL, (L-L_0))-np.dot(W_SM, (M-M_0))))
    print(np.sum(err_L**2)+np.sum(err_M**2)+np.sum(err_S**2))
    return np.sum(err_L**2)+np.sum(err_M**2)+np.sum(err_S**2)
        

# Test
data = pd.read_csv("example_data.csv")

x = data["age"].values
y = data["head"].values + 100

#x = np.array([1,2,3,4,3,4,5,5,2,3]) ## muss geordnet sein!
#y = np.array([2,3,4,1,3,4,2,2,2,3])

x_LMS = x_resample(x)
#
#L,M,S =init_params(x)
#
#z = z_transform(y, L, M, S)
#
#u_test = u_L(z, L, M, S)

'''
Start
'''

# Siehe Pub Cole 1992

alpha_M = len(x)*(max(x) - min(x))**3 /(400 * (max(y) - min(y))**2)
alpha_S = 2 * alpha_M * np.mean(x)
alpha_L = np.std(x)**4 * alpha_S
#alpha_M = 1
#alpha_S = 2
#alpha_L = 3

K = def_K(x_LMS)
#N = def_N(x_LMS)

L_0,M_0,S_0 = init_LMS_0(x_LMS)
L,M,S = init_LMS(x_LMS)
#L_update,M_update,S_update = init_LMS(x_LMS)

#LMS = np.zeros(len(x_LMS))

LL_0 = comp_LL(L, M, S, alpha_L, alpha_M, alpha_S, K, y, x, x_LMS)

diff_LL = 1 # percent


# outer loop
while diff_LL>0.04:
        
    diff_parameter = 1 # percent
    
    z = z_transform(y, x, x_LMS, L, M, S)
    
    u_L = (z/L)*(z-np.log(y/M)/S)- np.log(y/M)*(z**2 - 1)
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
    
    # inner loop
    # fehlerhaft
    while diff_parameter>0.04:
#        d = np.concatenate((L_0,M_0),axis=0)
#        d = np.concatenate((d,S_0),axis=0)
#        x0 = d
        
#        LMS_results = optimization.minimize(err_func, x0=[1,1,1,1,1,1], args=(L_0, M_0, S_0, y, x), method='Nelder-Mead')
#        
#        L_calc = LMS_results.x[0]* np.linspace(0, 1, len(x)) + LMS_results.x[1]
#        M_calc = LMS_results.x[2]* np.linspace(0, 1, len(x)) + LMS_results.x[3]
#        S_calc = LMS_results.x[4]* np.linspace(0, 1, len(x)) + LMS_results.x[5]
#        array_LMS = np.split(LMS_results, 3)
##        
#        L_calc = array_LMS[0]
#        M_calc = array_LMS[1]
#        S_calc = array_LMS[2]
        L_calc = np.dot(np.linalg.inv(W_L+alpha_L*K),(u_L+np.dot(W_L,L)-np.dot(W_LM, (M-M_0))-np.dot(W_LS, (S-S_0))))
        M_calc = np.dot(np.linalg.inv(W_M+alpha_M*K),(u_M+np.dot(W_M,M)-np.dot(W_MS, (S-S_0))-np.dot(W_ML, (L-L_0))))
        S_calc = np.dot(np.linalg.inv(W_S+alpha_S*K),(u_S+np.dot(W_S,S)-np.dot(W_SL, (L-L_0))-np.dot(W_SM, (M-M_0))))

        
        plt.plot(M)
        plt.plot(y, '.')
        
        plt.show()
        
        diff_parameter = max([np.sum(np.abs((L_0-L_calc)/L_calc))/len(L_calc), np.sum(np.abs((M_0-M_calc)/M_calc))/len(M_calc), np.sum(np.abs((S_0-S_calc)/S_calc))/len(S_calc)])
                 
        L_0 = L
        M_0 = M
        S_0 = S
        
        L =  L_calc
        M =  M_calc
        S =  S_calc
        print(diff_parameter)
        print("next")
        
#    M = M - min(M) + 1
#    S = np.linspace(0.1, 1, len(M))
#    L = np.linspace(0.1, 1, len(M))
    # hier klappt es nicht, weil M negative Werte aufweist
    LL_calc = comp_LL(L, M, S, alpha_L, alpha_M, alpha_S, K, y, x, x_LMS)
    diff_LL = np.abs((LL_0-LL_calc)/LL_calc)
    
    LL_0 = LL_calc
    print("LogLikelihood: " + str(diff_LL))

'''
trash
'''

#            
#            
#        L = L_update + L
#        M = M_update + M
#        S = S_update + S


        #print("diff = ") 
        #print(np.sum(np.abs((L_0-L)/L))/len(L), np.sum(np.abs((M_0-M)/M))/len(M), np.sum(np.abs((S_0-S)/S))/len(S))
#        print("M = ")        
#        print(M-M_0)
        # irgendwas ist falsch mit S oder M
        # der inner loop konvergiert nicht
#        plt.plot(L_update[-20:])
#        plt.show()
#        diff_parameter = max([np.sum(np.abs(L_update/L))/len(L), np.sum(np.abs(M_update/M))/len(M), np.sum(np.abs(S_update/S))/len(S)])
#        print(np.sum(np.abs(L_update/L)/len(L)), np.sum(np.abs(M_update/M))/len(M), np.sum(np.abs(S_update/S))/len(S))
#   

#        if np.sum(np.abs((L_0-L)/L))/len(L)>0.0001:
#            print(np.sum(np.abs((L_0-L)/L))/len(L))
#        if np.sum(np.abs((L_0-L)/L))/len(L)>0.0001:
#            print(np.sum(np.abs((L_0-L)/L))/len(L))
        #print("LMS change:" + str(diff_parameter))

#        test = [0]
#        for i in range(0, len(LMS)):
#            A_1 = [W_L[i]+alpha_L*np.diag(K)[i], W_LM[i], W_LS[i]]
#            A_2 = [W_ML[i], W_M[i]+alpha_M*np.diag(K)[i], W_MS[i]]
#            A_3 = [W_SL[i], W_SM[i], W_S[i]+alpha_S*np.diag(K)[i]]
#            
##            u_K_1 = u_L[i] - alpha_L * np.dot(K, L)[i]
##            u_K_2 = u_M[i] - alpha_M * np.dot(K, M)[i]
##            u_K_3 = u_S[i] - alpha_S * np.dot(K, S)[i]
#            
#            u_K_1 = u_L[i] - alpha_L * np.sum(K[i] * L)
#            u_K_2 = u_M[i] - alpha_M * np.sum(K[i] * M)
#            u_K_3 = u_S[i] - alpha_S * np.sum(K[i] * S)
#            
#            A = np.linalg.inv(np.matrix([A_1, A_2, A_3]))
#            
#            u_K = np.array([u_K_1, u_K_2, u_K_3])
#            LMS_update = np.dot(A, u_K)
#            
#            #was läuft falsch?
#            test = test + [u_K_2]
#            plt.plot(test)
#            plt.show()
#            
#            
#            L_update[i] = LMS_update[0,0]
#            M_update[i] = LMS_update[0,1]
#            S_update[i] = LMS_update[0,2]
            
#            plt.plot(L_update)
#            plt.show()
            
#        A = np.linalg.inv(np.matrix([A_1, A_2, A_3]))
#        u_K = np.array([u_K_1, u_K_2, u_K_3])
#            
#        LMS_update = np.dot(A, u_K)
            
#        L_update[i] = LMS_update[0,0]
#            M_update[i] = LMS_update[0,1]
#            S_update[i] = LMS_update[0,2]
            
        #print(L_update[-20:])
            
#            
#            
#        L = L_update + L
#        M = M_update + M
#        S = S_update + S


        #print("diff = ") 
        #print(np.sum(np.abs((L_0-L)/L))/len(L), np.sum(np.abs((M_0-M)/M))/len(M), np.sum(np.abs((S_0-S)/S))/len(S))
#        print("M = ")        
#        print(M-M_0)
        # irgendwas ist falsch mit S oder M
        # der inner loop konvergiert nicht
#        plt.plot(L_update[-20:])
#        plt.show()
#        diff_parameter = max([np.sum(np.abs(L_update/L))/len(L), np.sum(np.abs(M_update/M))/len(M), np.sum(np.abs(S_update/S))/len(S)])
#        print(np.sum(np.abs(L_update/L)/len(L)), np.sum(np.abs(M_update/M))/len(M), np.sum(np.abs(S_update/S))/len(S))
#        

'''
old
'''
#def u_L(z, L, M, S):
#    u = z/L*(z-np.log(y/M)/S)- np.log(y/M)*(z**2 - 1)
#    return u
#
#def u_M(z, L, M, S):
#    u = z / (M * S) + L * (z**2 - 1) / M
#    return u
#
#def u_S(z, S):
#    u = (z**2-1)/S
#    return u
#    
#def W_L(S):
#    W = 7*(S**2)/4
#    return np.diag(W)
#
#def W_M(L, M, S):
#    W = (1 + 2*(L**2)*(S**2))/M**2/S**2
#    return np.diag(W)
#
#def W_S(S):
#    W = 2/S**2
#    return np.diag(W)
#    
#def W_LM(M):
#    W = -1/(2*M)
#    return np.diag(W)
#
#def W_LS(L, M, S):
#    W = L * S
#    return np.diag(W)
#    
#def W_MS(L, M, S):
#    W = 2 * L / (M * S)
#    return np.diag(W)
######
#    
#def LL(params, x):
#    if (params[0]>0 and params[1]>0):
#        #prob = 0
#        prob_i = BCCG(params, x)
#        prob = np.sum(np.log(prob_i))
#        print(-prob)
#        return -prob
#    else:
#        return np.inf
#        
#def minimize_LL(x, initParams=None):
#    if initParams == None:
#        initParams = init_params(x)
#    else:
#        print(initParams)
#        
#    #results = minimize(LL, initParams, args=x, method='nelder-mead')
#    results = minimize(LL, initParams, args=x, method='bfgs')
#    #results = minimize(LL, initParams, args=x, method='L-BFGS-B')
#    return results
#    
#start = time.time()
#print("Start")    
#example_data = pd.read_csv("example_data.csv")
#x = example_data['head'].values
#
#
##initPar = [40, 0.1, 1]
#results = minimize_LL(x)
#
##Gewünschte Werte
##initParams = [48.33, 0.03432, 0.573, 1.389]
##results = minimize(LL, initParams, args=x, method='nelder-mead')
##results = minimize(LL, initParams, args=x, method='SLSQP')
##print(results.x)
#
#x_axis= np.arange(35,60,0.1)
#dist= BCCG(results.x,x_axis)
##dist= BCCG(results.x,x_axis)
#plt.plot(x_axis,dist,'r',label='BCCG') 
#plt.hist(x, bins='auto',normed=True)
#plt.legend()
#plt.title(str(results.x))
#plt.savefig("Python_BCCG_2.png")
#plt.show()
#
#end = time.time()
#print(str(end - start)+ " seconds")

