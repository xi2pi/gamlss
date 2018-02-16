# -*- coding: utf-8 -*-
"""
Created on Fri Feb  16  2018

@author: Christian Winkler
"""
import matplotlib.pyplot as plt
from scipy import exp
from scipy.special import gamma
import numpy as np
import pandas as pd
from scipy.optimize import minimize
#import time


    
def BCPE(M, S, L, T, x):
    c = np.sqrt(2**(-2/T)*gamma(1/T)/gamma(3/T))
    
    if L == 0:
        z = (1/S)*np.log(x/M)
    else:
        z = (1/(S*L))*(((x/M)**L)-1)
    
    f_z = T/(c*2**(1+1/T)*gamma(1/T))*exp(-0.5*np.abs(z/c)**T)
    f_y = (x**(L-1))/(M**L*S)*f_z
    return f_y

    
def LL(para, x, y):
    # compute the loglikelihood function

    Mcoeff = para[0:2]
    Scoeff = para[2] 
    Lcoeff = para[3]
    Tcoeff = para[4]
    
    polyM = np.poly1d(Mcoeff)
    Mp = np.polyval(polyM, x) 
        
    Mp_neg = Mp > 0
    if (Scoeff>0 and Tcoeff>0 and Mp_neg.all()):  
        prob_i = BCPE(Mp, Scoeff, Lcoeff, Tcoeff, y)
        prob = np.sum(np.log(prob_i))
        print(-prob)
        return -prob
    else:
        print("negativ")
        return np.inf
    
    # plotting for each iteration; can be taken out
    #plt.scatter(x, y)
    #plt.plot(x, Mp, '-', color= "r")
    #plt.show()      
    
        
def init_params(x, y):
    # initialize the parameters

    Mcoeff = np.polyfit(x, y, deg = 1)
#    Mcoeff = [1.9, 45]
#    Scoeff = 1.0
#    Lcoeff = 0.1
#    Tcoeff = 1.0
    #[  1.9925555   45.57392121   1.07551709   0.12995913   0.16770352]
#    Scoeff = 0.03
#    Lcoeff = 0.99
#    Tcoeff = 0.03
    #[  1.96795009e+00   4.54889361e+01   3.33270779e-02   1.025 1.28]
    # Similar to gamlss!!
    
    Scoeff = 0.1
    Lcoeff = 1.2
    Tcoeff = 0.1
    #[  1.94,   45.53,   0.0333, 1.04,   1.28]
    
    return [Mcoeff, Scoeff, Lcoeff, Tcoeff]
        
def minimize_LL(x, y):
    #Minimize the loglikelihood

    Mcoeff = init_params(x, y)[0]
    Scoeff = init_params(x, y)[1]
    Lcoeff = init_params(x, y)[2]
    Tcoeff = init_params(x, y)[3]
    #para = np.append(Mcoeff, Scoeff, Lcoeff, Tcoeff)
    para =list(Mcoeff)[0:2]+ [Scoeff]+ [Lcoeff]+ [Tcoeff]
    print(para)
    
    # Run minimizer
    results = minimize(LL, para, args=(x, y), method='nelder-mead')
    return results

# Load data
example_data = pd.read_csv("example_data.csv")
y = example_data['head'].values
x = example_data['age'].values
 
# Start optimizer
results = minimize_LL(x, y)

# Show results
x_axis= np.arange(min(x),max(x),0.01)

polyM = np.poly1d(results.x[0:2])
Mp = np.polyval(polyM, x_axis)
Sp = results.x[2]
Lp = results.x[3]
Tp = results.x[4]

# Computing percentiles
#y_axis= np.arange(min(y),max(y),0.01)

y_axis = [np.arange(Mp[i]-20, Mp[i]+20, 0.001) for i in range(0, len(x_axis))]

pdf = [BCPE(Mp[i], Sp, Lp, Tp, y_axis[i]) for i in range(0, len(x_axis))]
cdf = [np.cumsum(i) for i in pdf]

P04_id = [(np.abs(i/max(i) - 0.004)).argmin() for i in cdf]
P04 = [y_axis[i][P04_id[i]] for i in range(0, len(y_axis))]

P2_id = [(np.abs(i/max(i) - 0.02)).argmin() for i in cdf]
P2 = [y_axis[i][P2_id[i]] for i in range(0, len(y_axis))]

P10_id = [(np.abs(i/max(i) - 0.10)).argmin() for i in cdf]
P10 = [y_axis[i][P10_id[i]] for i in range(0, len(y_axis))]

P25_id = [(np.abs(i/max(i) - 0.25)).argmin() for i in cdf]
P25 = [y_axis[i][P25_id[i]] for i in range(0, len(y_axis))]

P75_id = [(np.abs(i/max(i) - 0.75)).argmin() for i in cdf]
P75 = [y_axis[i][P75_id[i]] for i in range(0, len(y_axis))]

P90_id = [(np.abs(i/max(i) - 0.9)).argmin() for i in cdf]
P90 = [y_axis[i][P90_id[i]] for i in range(0, len(y_axis))]

P98_id = [(np.abs(i/max(i) - 0.98)).argmin() for i in cdf]
P98 = [y_axis[i][P98_id[i]] for i in range(0, len(y_axis))]

P996_id = [(np.abs(i/max(i) - 0.996)).argmin() for i in cdf]
P996 = [y_axis[i][P996_id[i]] for i in range(0, len(y_axis))]


### Plotting scatterplt
plt.scatter(x, y)

### Plotting Percentiels
plt.plot(x_axis, Mp, '-', color= "r", label = "median")

plt.plot(x_axis, P996, '-', color= "y")
plt.plot(x_axis, P98, '-', color= "y")
plt.plot(x_axis, P90, '-', color= "y")
plt.plot(x_axis, P75, '-', color= "y")
#
plt.plot(x_axis, P25, '-', color= "y")
plt.plot(x_axis, P10, '-', color= "y")
plt.plot(x_axis, P2, '-', color= "y")
plt.plot(x_axis, P04, '-', color= "y")

plt.legend()
#plt.ylim([20, 80])
plt.savefig("Python_BCPE_2D.png")
plt.show()

###Parameter
#print(results.x[2:])

# GAMLSS
#> fitted(model_BCPE, "sigma")[1]
#      1231 
#0.03117349 
#> fitted(model_BCPE, "nu")[1]
#     1231 
#0.9884091 
#> fitted(model_BCPE, "tau")[1]
#      1231 
#0.03117349 

### Analysis of residuals
Mr = np.polyval(polyM, x)
np.mean(Mr-y)
np.std(Mr-y)
