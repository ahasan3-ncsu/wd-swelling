#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

jar = pd.read_csv('nominal_set.csv')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(jar['Grain Size (um)'],
           jar['Fission Density (10^21 f/cm3)'],
           jar['Fuel Gas Swelling (%)'])
plt.show()
