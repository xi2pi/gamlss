# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:44:30 2018

@author: Christian Winkler

Using the example from http://pythonhosted.org/fitter/
fitter package
"""

import pandas as pd
import time
from fitter import Fitter

    
example_data = pd.read_csv("example_data.csv")
x = example_data['head'].values

start = time.time()
print("Start")
#f = Fitter(x, distributions=['gamma'])
f = Fitter(x)
f.fit()
end = time.time()
print(str(end - start)+ " seconds")

# may take some time since by default, all distributions are tried
# but you call manually provide a smaller set of distributions
f.summary()



'''
Result
Fitting three distributions ['gamma', 'rayleigh', 'uniform'] takes 1.6 seconds

Later try with 'alpha'

'''