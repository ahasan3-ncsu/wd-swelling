import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test(fd, D):
    a1 = 2.53
    b1 = 0.606
    c1 = 4.27
    d1 = 0.45
    def A(D):
        return a1 * np.exp(-b1*(D-c1)) + d1

    a2 = 0.37
    b2 = 3.31
    c2 = 12.7
    k2 = 100
    def B(D):
        return c2 * (D/k2)**a2 * (1-D/k2)**b2

    def C(D):
        return 0.9

    a4 = 0.647
    b4 = 13.7
    c4 = 700
    k4 = 213.3
    def L(D):
        return c4 * (D/k4)**a4 * (1-D/k4)**b4

    a5 = 1.78
    b5 = 0.59
    c5 = 2.14
    d5 = 1.15
    def M(D):
        return a5 * np.exp(-b5*(D-c5)) + d5

    b6 = 2.04
    c6 = 2.95
    k6 = 0.24
    def N(D):
        return c6 * (b6 - np.exp(-k6*D))

    return A(D) / (1 + np.exp(-B(D) * (fd - C(D)))) \
        + L(D) / (1 + np.exp(-M(D) * (fd - N(D))))

vtest = np.vectorize(test)

X = np.linspace(0, 7, 50)
Y = np.linspace(1, 34, 100)
X, Y = np.meshgrid(X, Y)
Z = vtest(X, Y)


jar = pd.read_csv('nominal_set.csv')
fd = 'Fission Density (10^21 f/cm3)'
gr = 'Grain Size (um)'
fgs = 'Fuel Gas Swelling (%)'

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(X, Y, Z, cmap=plt.cm.jet, alpha=0.5,
                linewidth=0, antialiased=False)

ax.scatter(jar[fd], jar[gr], jar[fgs], c='black')
plt.show()
