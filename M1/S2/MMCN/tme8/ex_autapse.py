# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

# poids synaptique recurrent et l'entree du neurone
w = 4.23
x = 3 

# definition de fonctions
g = lambda a: 1./(1+np.exp(-a))

# parametres de la simulation
dt = 0.1
time = np.arange(0,10,dt)
T = len(time)
y = np.zeros(T)

# condition initiale pour y(t)
y[0] = 0

for t in range(1, T):

    # schema d'Euler pour dy/dt=-y+g(wy+x)
    y[t] =  (y[t] - y[t-1]) / dt

# initialisation de graphisme
plt.figure(1); plt.clf()

yy = np.arange( -0.5, 1.5, 0.01)

plt.subplot(311); plt.cla()
plt.plot(yy, g(w*yy+x), lw=3, label='f(y)=g(wy+x)')
plt.plot(yy, yy, lw=3, label='f(y)=y')
plt.axhline(0, ls=':', color = 'k')
plt.axvline(0, ls=':', color = 'k')
plt.xlabel('y')
plt.legend(loc='upper left')
plt.draw()

plt.subplot(312); plt.cla()
plt.plot(yy, -yy + g(w*yy+x), 'k-', lw=3, label='f(y)=-y+g(wy+x)')
plt.axhline(0, ls=':', color = 'k')
plt.axvline(0, ls=':', color = 'k')
plt.xlabel('y')
plt.legend()
plt.draw()

plt.subplot(313); plt.cla()
plt.plot(time, y, lw=3 )
plt.axhline(0, ls='-', color = 'k', lw=3)
plt.axvline(0, ls='-', color = 'k', lw=3)
plt.axhline(1, ls=':', color = 'k')
plt.xlabel('t')
plt.ylabel('y')
plt.xlim(-1,8)
plt.ylim(-0.1,1.1)
plt.draw()


plt.show(); 