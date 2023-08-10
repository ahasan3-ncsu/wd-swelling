import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pared(fd, D):
    a1 = 2.53
    b1 = 0.606
    c1 = 4.27
    d1 = 0.45
    def A(D):
        return a1 * np.exp(-b1*(D-c1)) + d1

    def B(D):
        return 3.08

    def C(D):
        return 0.932

    a4 = 38.431
    b4 = 0.09157
    c4 = 0.75734
    h4 = 17.9118
    def L(D):
        return a4 * np.exp(-b4*D) * np.log(c4*D) + h4

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


vpared = np.vectorize(pared)

X = np.linspace(0, 7, 50)
Y = np.linspace(1, 34, 100)
X, Y = np.meshgrid(X, Y)
Z = vpared(X, Y)


jar = pd.read_csv('nominal_set.csv')
fd = 'Fission Density (10^21 f/cm3)'
gr = 'Grain Size (um)'
fgs = 'Fuel Gas Swelling (%)'

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(X, Y, Z, cmap=plt.cm.jet, alpha=0.5,
                linewidth=0, antialiased=False)

ax.scatter(jar[fd], jar[gr], jar[fgs], c='black')

#ax.view_init(elev=10, azim=130)
ax.view_init(elev=10, azim=-40)
ax.set_xlabel(r'Fission density ($\times 10^{21}$ fission/cm$^3$)')
ax.set_ylabel('Grain size ($\mu m$)')
ax.set_zlabel('Fission gas swelling (%)')
plt.tight_layout()
#plt.savefig('img_6a.pdf')
plt.savefig('img_6b.pdf')
#plt.show()
