# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:20:58 2018

@author: Christian Winkler
"""
import matplotlib.pyplot as plt
from scipy import asarray as ar,exp
from scipy.special import erf, gamma
import numpy as np

def NO(x,M,S):
    return 1/(np.sqrt(2*np.pi)*S)*exp(-(x-M)**2/(2*S**2))

def BCCG(x,M,S,L):
    # Funktioniert noch nicht???
    # Musst noch überprüft werden!
    #Phi = 0.5*(1 + erf(((1/(S*np.abs(L)))-M)/(S*np.sqrt(2))))
    Phi = 0.5*(1 + erf((1/(S*np.abs(L)))/(np.sqrt(2))))
    
    if L == 0:
        z = (1/S)*np.log(x/M)
    else:
        z = (1/(S*L))*(((x/M)**L)-1)
    
    f = (x**(L-1)*np.exp(-0.5*z**2))/((M**L)*S*Phi*np.sqrt(2*np.pi))
    return f
    
def BCPE(x,M,S,L,T):
    c = np.sqrt(2**(-2/T)*gamma(1/T)/gamma(3/T))
    
    if L == 0:
        z = (1/S)*np.log(x/M)
    else:
        z = (1/(S*L))*(((x/M)**L)-1)
    
    f_z = T/(c*2**(1+1/T)*gamma(1/T))*exp(-0.5*np.abs(z/c)**T)
    f_y = (x**(L-1))/(M**L*S)*f_z
    return f_y

x = np.arange(0,20,0.1)

#plt.plot(x,NO(x,10,5),'b',label='Gaussian')
#plt.plot(x[1:],BCCG(x[1:],10,5,3),'r',label='BCCG')

#Smooth centile curves for skew and kurtotic data modelled using the Box-Cox power exponential distribution
#Rigby, Robert A., Stasinopoulos, D. Mikis

for i in [3, 2.5, 2, 1.5]:
    plt.plot(x[1:],BCPE(x[1:],10,0.1,3,i),'b',label='BCPE; T = ' + str(i) )
plt.legend()
plt.title('Distributions')
plt.show()

for i in [3, 2.5, 2, 1.5]:
    plt.plot(x[1:],BCPE(x[1:],10,0.1,-1,i),'b',label='BCPE; T = ' + str(i) )
    
plt.legend()
plt.title('Distributions')
plt.show()


# Modelling skewness and kurtosis with the BCPE density in GAMLSS

for i in [0.1, 0.4]:
    plt.plot(x[1:],BCPE(x[1:],5, i, 1, 1.2),'b',label='BCPE; M = ' + str(i) )
    
plt.legend()
plt.title('Distributions')
plt.show()
        
#####BCCG
#plt.plot(x[1:],BCCG(x[1:],5,0.5,-1),'b',label='BCCG; L = ' + str(-1) )
#    
#plt.legend()
#plt.title('Distributions')
#plt.show()




'''
function (mu.link = "identity", sigma.link = "log", nu.link = "identity") 
{
  mstats <- checklink("mu.link", "BC Cole Green", substitute(mu.link), 
    c("inverse", "log", "identity", "own"))
  dstats <- checklink("sigma.link", "BC Cole Green", substitute(sigma.link), 
    c("inverse", "log", "identity", "own"))
  vstats <- checklink("nu.link", "BC Cole Green", substitute(nu.link), 
    c("inverse", "log", "identity", "own"))
  
  structure(list(family = c("BCCG", "Box-Cox-Cole-Green"), 
    parameters = list(mu = TRUE, sigma = TRUE, nu = TRUE), 
    nopar = 3, type = "Continuous", mu.link = as.character(substitute(mu.link)), 
    sigma.link = as.character(substitute(sigma.link)), nu.link = as.character(substitute(nu.link)), 
    mu.linkfun = mstats$linkfun, sigma.linkfun = dstats$linkfun, 
    nu.linkfun = vstats$linkfun, mu.linkinv = mstats$linkinv, 
    sigma.linkinv = dstats$linkinv, nu.linkinv = vstats$linkinv, 
    mu.dr = mstats$mu.eta, sigma.dr = dstats$mu.eta, nu.dr = vstats$mu.eta, 
    
    dldm = function(y, mu, sigma, nu) {
      z <- ifelse(nu != 0, (((y/mu)^nu - 1)/(nu * sigma)), 
        log(y/mu)/sigma)
      dldm <- ((z/sigma) + nu * (z * z - 1))/mu
      dldm
      
    }, d2ldm2 = function(y, mu, sigma, nu) {
      d2ldm2 <- -(1 + 2 * nu * nu * sigma * sigma)
      d2ldm2 <- d2ldm2/(mu * mu * sigma * sigma)
      d2ldm2
      
    }, dldd = function(y, mu, sigma, nu) {
      z <- ifelse(nu != 0, (((y/mu)^nu - 1)/(nu * sigma)), 
        log(y/mu)/sigma)
      h <- dnorm(1/(sigma * abs(nu)))/pnorm(1/(sigma * 
        abs(nu)))
      dldd <- (z^2 - 1)/sigma + h/(sigma^2 * abs(nu))
      dldd
      
    }, d2ldd2 = function(sigma) {
      d2ldd2 <- -2/(sigma^2)
      d2ldd2
      
    }, dldv = function(y, mu, sigma, nu) {
      z <- ifelse(nu != 0, (((y/mu)^nu - 1)/(nu * sigma)), 
        log(y/mu)/sigma)
      h <- dnorm(1/(sigma * abs(nu)))/pnorm(1/(sigma * 
        abs(nu)))
      l <- log(y/mu)
      dldv <- (z - (l/sigma)) * (z/nu) - l * (z * z - 
        1)
      dldv <- dldv + sign(nu) * h/(sigma * nu^2)
      dldv
      
    }, d2ldv2 = function(sigma) {
      d2ldv2 <- -7 * sigma * sigma/4
      d2ldv2
      
    }, 
    d2ldmdd = function(mu, sigma, nu) -2 * nu/(mu * sigma), 
    d2ldmdv = function(mu) 1/(2 * mu), 
    d2ldddv = function(sigma, nu) -sigma * nu, 
    G.dev.incr = function(y, mu, sigma, nu, ...) -2 * dBCCG(y, mu = mu, sigma = sigma, nu = nu, 
      log = TRUE), 
    rqres = expression(rqres(pfun = "pBCCG", 
      type = "Continuous", y = y, mu = mu, sigma = sigma, 
      nu = nu)), mu.initial = expression(mu <- (y + mean(y))/2),
      
    sigma.initial = expression(sigma <- rep(0.1, length(y))), 
    nu.initial = expression(nu <- rep(0.5, length(y))), 
    mu.valid = function(mu) TRUE, sigma.valid = function(sigma) all(sigma > 
      0), nu.valid = function(nu) TRUE, y.valid = function(y) all(y > 
      0)), class = c("gamlss.family", "family"))
}
'''