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
#import scipy.stats as st
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
    #if (params[0]>0 and params[1]>0 and params[3]> 0):
    if (params[0]>0 and params[1]>0):
        prob = 0
        for i in x:
            prob_i = BCCG(params, i)
            prob = prob+np.log(prob_i)
        print(-prob)
        return -prob
    else:
        return np.inf

start = time.time()
print("Start")    
example_data = pd.read_csv("example_data.csv")


#x_2 = example_data['Unnamed: 0'].values
x = example_data['head'].values
#x = np.linspace(0, len(y_obs), num=len(y_obs)+1)

initParams = [40, 0.1, 1]


#Gewünschte Werte
#initParams = [48.33, 0.03432, 0.573, 1.389]
results = minimize(LL, initParams, args=x, method='nelder-mead')
#results = minimize(LL, initParams, args=x, method='SLSQP')
print(results.x)

x_axis= np.arange(35,60,0.1)
dist= BCCG(results.x,x_axis)
#dist= BCCG(results.x,x_axis)
plt.plot(x_axis,dist,'r',label='BCCG') 
plt.hist(x, bins='auto',normed=True)
plt.legend()
plt.title(str(results.x))
plt.savefig("Python_BCCG.png")
plt.show()

end = time.time()
print(str(end - start)+ " seconds")

'''
Results:
2.7 seconds BCCG

Found this page:
https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

There are 82 implemented distribution functions in SciPy 0.12.0. 
'''
