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


example_data = pd.read_csv("example_data.csv")
y = example_data['head'].values
X = example_data['age'].values

gam = LinearGAM(n_splines=4).fit(X, y) # your fitted model

# change resolution of X grid
XX = generate_X_grid(gam, n=20)

plt.figure(figsize=(10,8))
plt.scatter(X,y)
plt.plot(XX, gam.prediction_intervals(XX, quantiles=[.025,.5, .975]), color = "k")
plt.savefig("pygam_example_2.png")
plt.show()