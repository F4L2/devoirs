# -*-coding:Latin-1 -*
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# CREATION DE L'ENSEMBLE DES DONNEES (2 classes avec la distribution Gaussienne)
N = 100                                     # la taille de l'ensemble
m1x, m1y, s1 =   5.,  5., 1.                # moyenne (x,y) et largeur (s) de la classe 1
m2x, m2y, s2 =   1.,  1., 1.                # moyenne et largeur de la classe 2

c1x1 = np.random.normal(m1x, s1, N/2)       # coordonnees x des points dans la classe 1
c1x2 = np.random.normal(m1y, s1, N/2)       # coordonnees y des points
    
c2x1 = np.random.normal(m2x, s2, N/2)       # coordonnees x des points dans la classe 2
c2x2 = np.random.normal(m2y, s2, N/2)       # coordonnees y des points

datax1  = np.hstack([c1x1, c2x1])           # stocker les coordonees x des deux classes
datax2  = np.hstack([c1x2, c2x2])           # stocker les coordonees y des deux classes

labels = ['r']*(N/2) + ['b']*(N/2)          # etiquettes de points dans l'ensemble des donnees
                                            # 'r' (rouge) - classe 1, 'b' (bleu) - classe 2

index  = np.random.permutation(range(N))    # permuter aleatoirement les indices 
datax1  = datax1[index]                     # permuter les x    
datax2  = datax2[index]                     # permuter les y
labels  = [labels[i] for i in index]        # permuter les etiquettes 
                                            # (traitement special car 'labels' est un tableau des objets)

sigmoid   = lambda a: 1./(1. + np.exp(-a))           # definition d'une function sigmoide
sig_prime = lambda a: sigmoid(a)* (1. - sigmoid(a))  # derivee de la fonction sigmoide

# APPRENTISSAGE PAR LA DESCENTE DU GRADIENT
T = 10000                                    # nombre d'itérations
eta = 0.01                                   # taux d'apprentissage (learning rate)
w = [-1,1]                    # initialisation aleatoire des poids synaptiques

for i in range(T):
    
    p = np.random.randint(N)                # choix d'un indice aleatoire parmi N
    x = np.array([datax1[p], datax2[p]])    # motif  x(p)
    a = np.dot( w, x)                       # activation du neurone de sortie
    y = sigmoid(a)                          # activite du neurone de sortie

    # conversion des étiquettes à 0 et 1
    if labels[p] == 'r':
        target = 1.
    else: 
        target = 0.
        
    # delta rule
    delta = sig_prime(a) * ( y - target )   # erreur delta
    w = w - eta * delta * x                 # descente du gradient

      
# CLASSIFICATION
classe = []                      # initialisation du tableau pour garder les resultats de classification
for i in range(N):
    
    x = [datax1[i], datax2[i] ]             # i-eme motif 
    a = np.dot( w, x )                      # activation du neurone de sortie
    y = sigmoid(a)                          # activite du neurone de sortie

    # classification
    if y>0.5: 
        classe.append('r')
    else:
        classe.append('b')


# graphiques
plt.figure(1); plt.clf(); plt.show()                

# l'ensemble des donnees
plt.subplot(211)
plt.scatter( datax1, datax2, c = labels )            # donnees
plt.axhline(0, ls=':', color='k')                    # ligne horizontale pointillee (':') noire ('k')
plt.axvline(0, ls=':', color='k')                    # ligne verticale pointillee (':') noire ('k')
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('scaled')                                    
plt.title("L'ensemble des donnees")
plt.draw()

# resultat de classification 
plt.subplot(212)
plt.scatter( datax1, datax2, c = classe )
plt.axhline(0, ls=':', color='k')
plt.axvline(0, ls=':', color='k')
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('scaled')
plt.title('classification')
plt.draw()



                                        
