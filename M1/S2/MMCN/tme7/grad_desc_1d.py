# -*-coding:Latin-1 -*
import matplotlib.pyplot as plt
import numpy as np
    
eta =  0.2      # taux d'apprentissage
w   = -2.0      # valeur initiale de w
dw  =  0.0      # valeur initiale de dw

liste = []      # pour les graphiques 

for i in range(20):

    liste.append(w)    
    print('w = %.2f, dw = %.2f' % (w, dw ))

    dw = 1- 3*(w**2)
    w = w - eta * dw

print('Vrai minimum de E : %f' % (-1/np.sqrt(3)) )    
print('Minimum par descente du gradient : %f' % w )    

# figures
plt.figure(1); plt.clf();             

plt.subplot(211)
x = np.arange(-2,2,0.01)
plt.plot(x, x - x**3 )
plt.axhline(0, ls=':', color='k')   # ligne horizontale pointillee (':') noire ('k')
plt.axvline(0, ls=':', color='k')   # ligne verticale pointillee (':') noire ('k')
plt.axvline(-1/np.sqrt(3), color='r')   # minimum de E
plt.xlabel("w")
plt.ylabel("E")
plt.draw()

plt.subplot(212)
plt.plot(liste, 'o-')
plt.axhline(-1/np.sqrt(3), color='r')
plt.xlabel("iterations")
plt.ylabel("w")
plt.draw()

plt.show() 