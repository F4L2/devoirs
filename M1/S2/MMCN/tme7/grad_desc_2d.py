# -*-coding:Latin-1 -*
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def E(x,y):
    return 4*(x-1)**2 + 5*(y-2)**2


eta = 0.2                       # taux d'apprentissage
w  = np.array([3., 1.])         # valeur initiale de w
dw = np.array([0., 0.])         

liste = []

for i in range(10):

    liste.append(w)    
    print('dw = (%.2f, %.2f), w = (%.2f, %.2f)' % (dw[0], dw[1], w[0], w[1]) )
    
    dw[0], dw[1] = 8*(w[0]-1) , 10*(w[1]-2)
    w = w[:]-eta*dw[:]

print('Vrai minimum de E : (%f,%f)' % (1,2) )    
print('Minimum par descente du gradient : (%f,%f)' % (w[0],w[1]) )    


# figures
fig = plt.figure(1); plt.clf();              

# E(w1,w2)
ax = fig.add_subplot(211, projection='3d')
w1 = np.arange(-10,10,0.1)
w2 = np.arange(-10,10,0.1)
W1, W2 = np.meshgrid(w1,w2)
ax.plot_surface(W1,W2,E(W1, W2), cmap='jet')
plt.xlabel("w1")
plt.ylabel("w2")
plt.title("E(w1,w2)")
plt.draw()

# descent du gradient
plt.subplot(212)
liste = np.array(liste)
plt.plot( liste[:,0], liste[:,1], 'o-' )
plt.axvline(0, ls=':', color='k')
plt.axhline(0, ls=':', color='k')
plt.xlabel("w1")
plt.ylabel("w2")
plt.axis('scaled')
plt.xlim(-1,5)
plt.ylim(-1,5)
plt.title('Descent du gradient')
plt.draw()

plt.show()  