# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:20:58 2018

@author: Christian Winkler
"""
import matplotlib.pyplot as plt
from scipy import exp
#from scipy.special import gamma, erf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
#import time


def NO(S, x): 
    # normal distribution
    return 1/(np.sqrt(2*np.pi)*S)*exp(-(x)**2/(2*S**2))
    
    
def LL(para, x, y):
    # compute the loglikelihood function

    Mcoeff = para[0:4]
    Scoeff = para[4] 
    
    polyM = np.poly1d(Mcoeff)
    Mp = np.polyval(polyM, x) 
    
    Sp = Scoeff
    
    prob = 0
    for i in range(0, len(y)):
        #print(i)
        prob_i = NO(Sp, y[i]-Mp[i])
        if prob_i > 0:
            prob = prob+np.log(prob_i)
            print(-prob)
        else:
            print("negativ")
    
    # plotting for each iteration; can be taken out
    plt.scatter(x, y)
    plt.plot(x, Mp, '-', color= "r")
    plt.show()      
    
    return -prob
        
        
def init_params(x, y):
    # initialize the parameters

    Mcoeff = np.polyfit(x, y, deg = 3)
    #Mcoeff = np.array([1,2,3,40])
    Scoeff = 1.0
    return [Mcoeff, Scoeff]
        
def minimize_LL(x, y):
    #Minimize the loglikelihood

    Mcoeff = init_params(x, y)[0]
    Scoeff = init_params(x, y)[1]
    para = np.append(Mcoeff, Scoeff)
    
    # Run minimizer
    results = minimize(LL, para, args=(x, y), method='nelder-mead')
    return results


example_data = pd.read_csv("example_data.csv")
y = example_data['head'].values
x = example_data['age'].values
 

results = minimize_LL(x, y)

x_axis= np.arange(min(x),max(x),0.1)

polyM = np.poly1d(results.x[0:4])
Mp = np.polyval(polyM, x_axis)
Sp = results.x[4]


plt.scatter(x, y)
plt.plot(x_axis, Mp, '-', color= "r")
plt.plot(x_axis, Mp+Sp, '-', color= "r")
plt.plot(x_axis, Mp-Sp, '-', color= "r")
plt.show()




