import numpy as np
import matplotlib.pyplot as plt

def LJ126(r, eps=1.0, rmin=1.0, rc=2.5):
    if r <= rc:
        return 4*eps*((rmin/r)**12 - (rmin/r)**6)   # LJ 12-6
    else:
        return 0.0


def FENE(r, k=30, r0=1.8):
    if r <= r0:
        return -0.5*k*r0**2*np.log(1-(r/r0)**2)
    else:
        return 0.0
lj_vec = np.vectorize(LJ126)
FENE_vec = np.vectorize(FENE)



r = np.linspace(0.7,2.6)
rmin = 1.0
eps = 1.0
ylj = lj_vec(r, eps, rmin, rc=1.0)
yfene = FENE_vec(r, k=30, r0=1.8)
plt.plot(r,ylj,'-r',label='lj')
plt.plot(r,yfene,'-b',label='FENE')
plt.legend()
plt.title("Effective U for bonded particles")
plt.show()


r = np.linspace(0.9,3.0)
rmin = 1.0
eps = 1.0
ylj = lj_vec(r, eps, rmin=2.5, rc=2.5)
plt.plot(r,ylj,'-b',label='lj, rmin=2.5')
plt.ylim(-1.1,3)
plt.legend()
plt.title("Effective U for Non-bonded particles")
plt.show()


r = np.linspace(0.9,10.0)
rmin = 1.0
eps = 1.0
ylj = lj_vec(r, eps, rmin=3.75, rc=9.375)
plt.plot(r,ylj,'-b',label='lj, rmin=2.5')
plt.ylim(-1.1,3)
plt.legend()
plt.title("Effective U for Non-bonded particles")
plt.show()