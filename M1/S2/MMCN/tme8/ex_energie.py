# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# poids synaptique recurrent et l'entree du neurone
w = 10.
x = -w/2

# fonction d'activation
g = lambda a: 1./(1+np.exp(-a))

# fonction energie
# E = lambda y: 

# parametres de la simulation
dt = 0.1
time = np.arange(0,10,dt)
T = len(time)
y = np.zeros(T)

# condition initiale pour y(t)
y[0] = 0.6

# initialisation du graphisme
plt.figure(1); plt.show(); plt.clf()

for t in range(1, T):

    print('Iteration %d' % t)
    
    # methode d'Euler pour dy/dt=-y+g(wy+x)
    y[t] = y[t-1]+ dt*(-y[t-1] + g(w*y[t-1]+x)) 

    # fonction energie
    plt.subplot(211); plt.cla()
    yy = np.arange( -0.5, 1.5, 0.01)
    plt.plot(yy, E(yy), 'k-', lw=3 )    # fonction energie
    plt.plot( [y[t]], [E(y[t])], 'ro', markersize=10)  # energie de l'etat actuel
    plt.axhline(0, ls=':', color = 'k')
    plt.axvline(0, ls=':', color = 'k')
    plt.draw()
    
    # solution
    plt.subplot(212); plt.cla()
    plt.plot(time[:t+1], y[:t+1], lw=3 )
    plt.axhline(0, ls='-', color = 'k', lw=3)
    plt.axvline(0, ls='-', color = 'k', lw=3)
    plt.axhline(1, ls=':', color = 'k', lw=2)
    plt.axhline(0.5, ls=':', color = 'k', lw=2)
    plt.xlim(-1,8)
    plt.ylim(-0.1,1.1)
    plt.draw()
    
    sleep(0.1)

