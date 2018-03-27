# -*- coding: utf-8 -*-
"""
@author: Christian Winkler
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pygam import LinearGAM
import pygam

from pygam.utils import generate_X_grid


example_data = pd.read_csv("example_db_dataset.csv")
y = example_data['head'].values
X = example_data['age'].values

gam =  pygam.GAM(n_splines=15).gridsearch(X, y)
#, distribution='gamma'

gam.fit(X, y) # your fitted model

# change resolution of X grid
XX = generate_X_grid(gam, n=100)
samples = gam.sample(X, y, quantity='y', n_draws=50000, sample_at_X=XX)

# sampels is shape (len(XX), len(y))

percentiles = np.percentile(samples, q=[2.5, 97.5, 50.0], axis=0)
#percentiles is now shape (2, len(y))

plt.figure(figsize=(10,8))
plt.scatter(X, y, s = 1)
plt.plot(XX, percentiles[0], "k")
plt.plot(XX, percentiles[1], "k")
plt.plot(XX, percentiles[2], "k")
plt.savefig("pygam_example_3.png", dpi = 900)
plt.show()