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

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(jar['fission_density'], jar['fuel_temp'], jar['fuel_gas_swelling'])
#plt.show()

def fun(x, l, k, x0):
    return l / (1 + np.exp(-k * (x-x0)))

def f2(x, a, b):
    fd, D = x
    return fun(fd, 42.9, 1.13, 5.47) * (a * D + b)

def f3(x, c, d, e):
    fd, D, r, T = x
    return f2((fd, D), -0.0527, 1.54) * (c * r + d * T + e)

def f4(x, p, q):
    fd, D, r, T = x
    return f3((fd, D, r), 0.12, 0.43) * (p * T + q)

#popt, pcov = curve_fit(fun, jar['fission_density'], jar['fuel_gas_swelling'],
#        bounds=([0,0,0], [np.inf,np.inf,np.inf]))
#print(popt)

popt, pcov = curve_fit(f3,
        (jar['fission_density'], jar['grain_size'], jar['fission_rate'], jar['fuel_temp']),
        jar['fuel_gas_swelling'])
        #bounds=([0,0,0], [np.inf,np.inf,np.inf]))
print(popt)

y_pred = f3((tra['fission_density'], tra['grain_size'],
    tra['fission_rate'], tra['fuel_temp']), *popt)
plt.plot(tra['fission_density'], tra['fuel_gas_swelling'], label='data')
plt.plot(tra['fission_density'], y_pred, label='func')
plt.legend()
plt.show()
