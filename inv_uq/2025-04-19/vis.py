import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

jar = pd.read_csv('normal.csv')

fdVals = jar['FD'].unique()
hi_fd = jar[jar['FD'] == fdVals[-2]]
hi_fd = hi_fd.drop(['FD'], axis=1)

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
    plt.scatter(hi_fd[columns[i]], hi_fd[columns[9]], s=5,
                c=hi_fd[columns[0]], cmap='viridis')
    plt.xlabel(columns[i])
    plt.ylabel(columns[9])
    #plt.legend()
    plt.colorbar()
    plt.show()

# plt.scatter(jar[columns[7]], jar[columns[8]] - jar[columns[9]])
#
# x = np.linspace(1e21, 8e21, num=8)
# y = 4e-21 * x
#
# plt.plot(x, y, c='r')
#
# plt.show()
