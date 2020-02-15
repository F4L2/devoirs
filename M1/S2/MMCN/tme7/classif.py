# -*-coding:Latin-1 -*
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# CREATION DE L'ENSEMBLE DES DONNEES 
#
# creer un ensemble de donnees (2 classes en 2D)
# 
N = 100
c1x1 = np.random.normal(4, 1, int(N/2))
c1x2 = np.random.normal(4, 1, int(N/2))

c2x1 = np.random.normal(0, 1, int(N/2))
c2x2 = np.random.normal(0, 1, int(N/2))

datax1  = np.hstack([c1x1, c2x1]) 
datax2  = np.hstack([c1x2, c2x2])  
labels = ['r']*int(N/2) + ['b']*int(N/2)

def sigmoid(a):
    return 1./(1 + np.exp(-a))   # definition d'une function sigmoide

# initialiser le vecteur des poids 
w = [1,1]
bias = -4

# CLASSIFICAITON (sans apprentissage)
classe = []          # pour sauvegarder les resultats de classification

for i in range(N):
    
    motif = [datax1[i], datax2[i]]              # i-eme motif 
    a     = np.dot( w, motif ) +bias                 # activation du neurone de sortie
    y     = sigmoid(a)                          # activite du neurone de sortie

    # classification
    if y>0.5: 
        classe.append('r')
    else:
        classe.append('b')


# figures
plt.figure(1); plt.clf();                

# l'ensemble des donnees
plt.subplot(221)
plt.scatter( datax1, datax2, c = labels )            # donnees
plt.axhline(0, ls=':', color='k')                    # ligne horizontale pointillee (':') noire ('k')
plt.axvline(0, ls=':', color='k')                    # ligne verticale pointillee (':') noire ('k')
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('scaled')                                    
plt.title("L'ensemble des donnees")
plt.draw()

# resultat de classification (sans apprentissage)
plt.subplot(222)
plt.scatter( datax1, datax2, c = classe )
plt.arrow( 0, 0, w[0], w[1]) # vecteur normal
ax1 = -2
ax2 = -bias - w[0]/w[1]*ax1 
bx1 = 2
bx2 = -bias - w[0]/w[1]*bx1 
plt.plot([ax1, bx1], [ ax2, bx2], 'g', lw=2 )                 # hyperplan separateur
plt.axhline(0, ls=':', color='k')
plt.axvline(0, ls=':', color='k')
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('scaled')
plt.title('classification')
plt.draw()

# fonction d'activation (sigmoide)
plt.subplot(223)
a = np.arange(-10,10,0.1)
plt.plot( a, sigmoid(a) )
plt.axhline(0.5, ls='--', color='r')
plt.xlabel("a")
plt.ylabel("sigmoid(a)")
plt.title ("fonction d'activation g(a)")
plt.draw()

# VISUALISATION DE Y(X1,X2)
x1range  = np.arange(-10,10,0.1)
x2range  = np.arange(-10,10,0.1)
y = np.zeros(( len(x2range), len(x1range) ))
for i, x2 in enumerate(x2range):
    for j, x1 in enumerate(x1range):
        
        x       = [x1, x2]                  # vecteur des entrees
        a       = np.dot( w, x)             # activation du neurone de sortie
        y[i,j]  = sigmoid(a)                # activite du neurone de sortie

# figure : visualisation de y(x1,x2)
X1, X2 = np.meshgrid(x1range, x2range)
ax = plt.subplot(224, projection='3d')
ax.plot_surface(X1, X2, y, cmap='jet')
plt.xlabel("x1")
plt.ylabel("x2")
ax.set_zlabel("y")
plt.title("y(x1,x2)")
plt.draw()


plt.show() 