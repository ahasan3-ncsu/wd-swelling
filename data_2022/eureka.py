import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sig(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# the nominal set
jar = pd.read_csv('nominal_set.csv')
fd = 'Fission Density (10^21 f/cm3)'
gr = 'Grain Size (um)'
fgs = 'Fuel Gas Swelling (%)'

for G in [1.34, 4.36, 8.5, 17, 34]:
    # select grain size
    tra = jar.loc[jar[gr] == G]

    # cutoff based on local minimum
    dd = np.diff(tra[fgs])
    pivlo = len(dd) // 7 * 2
    pivhi = pivlo * 2
    mi = pivlo + np.argmin(dd[pivlo:pivhi])
    cut = tra[fgs].iloc[mi]
    print(mi, mi/132*7)
    print(cut)

    low = tra.loc[tra[fgs] <= cut]
    high = tra.loc[tra[fgs] > cut]
    high[fgs] = high[fgs] - cut

    p1, c1 = curve_fit(sig, low[fd], low[fgs],
                       bounds=([0,0,0], [np.inf,np.inf,np.inf]))
    p2, c2 = curve_fit(sig, high[fd], high[fgs],
                       bounds=([0,0,0], [np.inf,np.inf,np.inf]))
    print(p1, p2)

    pred = [sig(x, *p1) + sig(x, *p2) for x in tra[fd]]
    plt.plot(tra[fd], tra[fgs])
    plt.plot(tra[fd], pred, ls='-.')

plt.show()
