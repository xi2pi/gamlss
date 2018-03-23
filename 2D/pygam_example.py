# -*- coding: utf-8 -*-
"""
@author: Christian Winkler
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pygam import LinearGAM
import pygam


example_data = pd.read_csv("example_data.csv")
y = example_data['head'].values
X = example_data['age'].values

#gam =  LinearGAM()
#gam =  LinearGAM(n_splines=10).gridsearch(X, y)
gam =  pygam.GAM(n_splines=10).gridsearch(X, y)

gam.fit(X, y) # your fitted model
samples = gam.sample(X, y, quantity='y', n_draws=500, sample_at_X=X)
# sampels is shape (500, len(y))

percentiles = np.percentile(samples, q=[2.5, 97.5], axis=0)
#percentiles is now shape (2, len(y))


plt.figure(figsize=(10,8))
plt.scatter(X,y)
plt.plot(X,percentiles[0])
plt.plot(X,percentiles[1])
# plotting

plt.savefig("pygam_example.png")
plt.show()

