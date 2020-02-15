# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

# generation de donnees (deux nuages gaussiens)
N = 100                                          
m1x1, m1x2, s1 =  -1,  5., 1. 
m2x1, m2x2, s2 =   5., 1., 1.

half = int(N/2)
c1x1 = np.random.normal(m1x1, s1, half)               
c1x2 = np.random.normal(m1x2, s1, half)               

c2x1 = np.random.normal(m2x1, s2, half)               
c2x2 = np.random.normal(m2x2, s2, half)               

x1  = np.hstack([c1x1, c2x1])             # stocker les coordonees x des deux classes 
x2  = np.hstack([c1x2, c2x2])             # stocker les coordonees y des deux classes

# parametres de simulation
T = 10000
eta = 0.001

# initisalisation du vecteur de poids
W = np.random.rand(2,2) -0.5


plt.figure(1); plt.clf();    # initialisation graphique

for i in range(T):
    
    p = np.random.randint(N)
    x = np.array([x1[p], x2[p]])
    y = np.dot(W, x)

    # ajouter l'apprentissage competitif pour la matrice de poids W
    # afin de trouver l'index de la sortie maximale, utiliser np.argmax(y)
    # ...
    y_win = np.argmax(y)
    y = y * 0
    y[y_win] = 1

    oja_rule = eta*y*(x-y*W)

    print(oja_rule)
    W = W + oja_rule

    if i%100==0:

        print('Iteration %d' % i)
        plt.subplot(111); plt.cla()
        plt.scatter( x1, x2 )  
        plt.plot([0, W[0,0]], [0, W[0,1]], '-r', lw=3)
        plt.plot([0, W[1,0]], [0, W[1,1]], '-g', lw=3)
        plt.axhline(0, ls=':', color='k')                    
        plt.axvline(0, ls=':', color='k')                    
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.axis('scaled')                                    
        plt.draw()

        plt.pause(0.1)

## nouvelles donnees a classifier
# N = 100
# nouvelles_donnees = -5 + np.random.random_sample((100,2))*10
# 
# # classification non-supervise
# classes = []
# for x in nouvelles_donnees:
#     y = np.dot(W,x)
#     if y[1] > y[0] : classes.append('g')    # classe 1
#     else : classes.append('r')              # classe 2
# 
# 
# plt.subplot(212); plt.cla()
# plt.scatter( nouvelles_donnees[:,0], nouvelles_donnees[:,1], color=classes )  
# plt.axhline(0, ls=':', color='k')                    
# plt.axvline(0, ls=':', color='k')                    
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.axis('scaled')                                    
# plt.draw()


plt.show()