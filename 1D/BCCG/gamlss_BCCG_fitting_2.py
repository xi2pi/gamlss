# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:20:58 2018

@author: Christian Winkler
"""
import matplotlib.pyplot as plt
from scipy import exp
from scipy.special import gamma, erf
import numpy as np
import pandas as pd
#from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.optimize import minimize
import time

def NO(x,M,S):
    return 1/(np.sqrt(2*np.pi)*S)*exp(-(x-M)**2/(2*S**2))
    
def BCCG(params,x):
    M = params[0]
    S = params[1]
    L = params[2]
    
    Phi = 0.5*(1 + erf((1/(S*np.abs(L)))/(np.sqrt(2))))
    
    if L == 0:
        z = (1/S)*np.log(x/M)
    else:
        z = (1/(S*L))*(((x/M)**L)-1)
    
    f = (x**(L-1)*np.exp(-0.5*z**2))/((M**L)*S*Phi*np.sqrt(2*np.pi))
    return f

def BCPE(params, x):
    M = params[0]
    S = params[1]
    L = params[2]
    T = params[3]
    c = np.sqrt(2**(-2/T)*gamma(1/T)/gamma(3/T))
    
    if L == 0:
        z = (1/S)*np.log(x/M)
    else:
        z = (1/(S*L))*(((x/M)**L)-1)
    
    f_z = T/(c*2**(1+1/T)*gamma(1/T))*exp(-0.5*np.abs(z/c)**T)
    f_y = (x**(L-1))/(M**L*S)*f_z
    return f_y
    

def LL(params, x):
    if (params[0]>0 and params[1]>0):
        #prob = 0
        prob_i = BCCG(params, x)
        prob = np.sum(np.log(prob_i))
        print(-prob)
        return -prob
    else:
        return np.inf
        
def init_params(x):
    M = np.mean(x)
    S = np.std(x)
    L = stats.skew(x)
    return [M, S, L]
        
def minimize_LL(x, initParams=None):
    if initParams == None:
        initParams = init_params(x)
    else:
        print(initParams)
        
    #results = minimize(LL, initParams, args=x, method='nelder-mead')
    results = minimize(LL, initParams, args=x, method='bfgs')
    #results = minimize(LL, initParams, args=x, method='L-BFGS-B')
    return results
    
start = time.time()
print("Start")    
example_data = pd.read_csv("example_data.csv")
x = example_data['head'].values


#initPar = [40, 0.1, 1]
results = minimize_LL(x)

#Gewünschte Werte
#initParams = [48.33, 0.03432, 0.573, 1.389]
#results = minimize(LL, initParams, args=x, method='nelder-mead')
#results = minimize(LL, initParams, args=x, method='SLSQP')
#print(results.x)

x_axis= np.arange(35,60,0.1)
dist= BCCG(results.x,x_axis)
#dist= BCCG(results.x,x_axis)
plt.plot(x_axis,dist,'r',label='BCCG') 
plt.hist(x, bins='auto',normed=True)
plt.legend()
plt.title(str(results.x))
plt.savefig("Python_BCCG_2.png")
plt.show()

end = time.time()
print(str(end - start)+ " seconds")

'''
Results:
nelder- mead: 2.3 seconds BCCG with init parameters
[  4.83247539e+01,   3.57880757e-02,   4.21610581e-01]


Update 16.02.18: Tipp by statsmodel author: change "i in x"
0.46 seconds

bfgs: 0.47 seconds; small differences in parameters

Found this page:
https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

There are 82 implemented distribution functions in SciPy 0.12.0. 
'''
