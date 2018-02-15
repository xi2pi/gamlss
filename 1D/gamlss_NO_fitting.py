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
#import scipy.stats as stats
from scipy.optimize import minimize
import time

def NO(params, x):
    M = params[0]
    S = params[1]
    
    return 1/(np.sqrt(2*np.pi)*S)*exp(-(x-M)**2/(2*S**2))
    
    

def LL(params, x):
    if (params[0]>0 and params[1]>0):
        prob = 0
        for i in x:
            prob_i = NO(params, i)
            prob = prob+np.log(prob_i)
        print(-prob)
        return -prob
    else:
        return np.inf
        
def init_params(x):
    M = np.mean(x)
    S = np.std(x)
    return [M, S]
        
def minimize_LL(x, initParams=None):
    if initParams == None:
        initParams = init_params(x)
    else:
        print(initParams)
        
    results = minimize(LL, initParams, args=x, method='nelder-mead')
    return results
    
start = time.time()
print("Start")    
example_data = pd.read_csv("example_data.csv")
x = example_data['head'].values


#initPar = [40, 0.1, 1]
results = minimize_LL(x)




'''
Plotting
'''
x_axis= np.arange(35,60,0.1)
dist= NO(results.x,x_axis)
#dist= BCCG(results.x,x_axis)
plt.plot(x_axis,dist,'r',label='NO') 
plt.hist(x, bins='auto',normed=True)
plt.legend()
plt.title(str(results.x))
plt.savefig("Python_NO.png")
plt.show()

end = time.time()
print(str(end - start)+ " seconds")

'''
Results:


Found this page:
https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

There are 82 implemented distribution functions in SciPy 0.12.0. 
'''
