# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:20:58 2018

@author: Christian Winkler
"""
import matplotlib.pyplot as plt
from scipy import asarray as ar,exp
from scipy.special import erf, gamma
import numpy as np

def BCPE(M, S, L, T, x):
    c = np.sqrt(2**(-2/T)*gamma(1/T)/gamma(3/T))
    
    if L == 0:
        z = (1/S)*np.log(x/M)
    else:
        z = (1/(S*L))*(((x/M)**L)-1)
    
    f_z = T/(c*2**(1+1/T)*gamma(1/T))*exp(-0.5*np.abs(z/c)**T)
    f_y = (x**(L-1))/(M**L*S)*f_z
    return f_y
    

x = np.arange(40,60,0.1)

#plt.plot(x,NO(x,10,5),'b',label='Gaussian')
#plt.plot(x[1:],BCCG(x[1:],10,5,3),'r',label='BCCG')

#Smooth centile curves for skew and kurtotic data modelled using the Box-Cox power exponential distribution
#Rigby, Robert A., Stasinopoulos, D. Mikis
y= BCPE(47.48, 0.031, 0.99, 1.29, x)

y_cum = np.cumsum(y)

plt.plot(x,y,'b',label='BCPE' )
plt.legend()
plt.title('Distributions')
plt.show()


# cumulative distribution
plt.plot(x,y_cum/max(y_cum),'b',label='BCPE' )
plt.legend()
plt.title('Cum. Distribution')
plt.show()


# Percentiles
idx = (np.abs(y_cum/max(y_cum) - 0.02)).argmin()
x[idx]

