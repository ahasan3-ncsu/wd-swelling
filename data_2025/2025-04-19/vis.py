import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

jar = pd.read_csv('normal.csv')

columns = ['dGrainHBS',
           'FaceCovMax',
           'SwellLink',
           'vResol',
           'DatomFissGBx',
           'fNucleate',
           'aAtomDifFiss',
           'FD',
           'fuel_swelling',
           'fuel_gas_swelling']

for i in range(7):
    plt.scatter(jar[columns[8]], jar[columns[9]],
                c=jar[columns[i]], cmap='viridis')
    plt.show()

# plt.scatter(jar[columns[7]], jar[columns[8]] - jar[columns[9]])
#
# x = np.linspace(1e21, 8e21, num=8)
# y = 4e-21 * x
#
# plt.plot(x, y, c='r')
#
# plt.show()
