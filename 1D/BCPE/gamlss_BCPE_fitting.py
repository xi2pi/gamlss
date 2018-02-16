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
    if (params[0]>0 and params[1]>0 and params[3]> 0):
        prob = 0
        prob_i = BCPE(params, x)
        prob = np.sum(np.log(prob_i))
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

initParams = [40, 0.1, 1, 1.7]



#results = minimize(LL, initParams, args=x, method='nelder-mead')
results = minimize(LL, initParams, args=x, method='bfgs')
print(results.x)

x_axis= np.arange(35,60,0.1)
dist= BCPE(results.x,x_axis)
#dist= BCCG(results.x,x_axis)
plt.plot(x_axis,dist,'r',label='BCPE') 
plt.hist(x, bins='auto',normed=True)
plt.legend()
plt.title(str(results.x))
plt.savefig("Python_BCPE.png")
plt.show()

end = time.time()
print(str(end - start)+ " seconds")

'''
Results:
Nelder-Mead kommt auf 4.3 seconds
[  4.83366348e+01   3.53230980e-02   7.92914982e-01   1.38759560e+00]

bfgs gibt falsche parameter

L-BFGS-B ebenfalls falsche parameter



#Gewünschte Werte (aus R/gamlss)
#initParams = [48.33, 0.03432, 0.573, 1.389]

Found this page:
https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

There are 82 implemented distribution functions in SciPy 0.12.0. 
'''
