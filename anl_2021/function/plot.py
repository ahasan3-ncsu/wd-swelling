#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

jar = pd.read_csv('swelling.csv')
#tra = jar.truncate(after=409)
tra = jar[2651:2786]

#plt.figure(figsize=(8,6))
#plt.scatter(jar['fuel_temp'], jar['fuel_swelling'], label='fuel swelling')
#plt.scatter(jar['fuel_temp'], jar['fuel_gas_swelling'], label='fuel gas swelling')
#plt.xlabel('Temperature')
#plt.ylabel('Percentage')
#plt.legend()
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(jar['grain_size'], jar['fission_density'], jar['fuel_gas_swelling'])
plt.show()
