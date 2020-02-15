# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# donnees
donnees = np.array([[ 1., 1.],  [ 3., 1.], [-2., 0.], [-2.,-2.]]) 
N = len(donnees)

# generer des donnees avec una matrice de covariance donnee
N = 50
moyenne = np.array([10.0, 10.0])    #si l'on est pas au centre, le vecteur s'aligne avec le centre du nuage de point
C = np.array([ [ 4.5, 2], [ 2, 1.5] ])
donnees = np.random.multivariate_normal(moyenne, C, size=N)

x1 = donnees[:,0]
x2 = donnees[:,1]

# calculer la matrice de covariance
C = donnees.T.dot(donnees) / len(donnees)

# valeurs et vecteurs propres de norme 1
valp, vecp =  np.linalg.eig(C)
print('Valeurs propres de la matrice de covariance: %.1f, %.1f' % (valp[0], valp[1]) )

# l'axe principal
ind_max = np.argmax( valp)      # l'index de la valeur propre maximale
axe_princ = vecp[:, ind_max]    # le vecteur propre correspondant


plt.figure(1); plt.clf(); 

plt.subplot(211)
plt.scatter( x1, x2, c='b' )  
plt.arrow( 0, 0, axe_princ[0], axe_princ[1], color='r', lw=2)
plt.axhline(0, ls=':', color='k')                    
plt.axvline(0, ls=':', color='k')                    
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('scaled')                                    
plt.draw()

input("\n Press any key \n")

# reseau avec l'apprentissage Hebbien
T = 100
eta = 0.01
w = np.random.rand(2) - 0.5

for i in range(T):
    
    p = np.random.randint(N)
    x = donnees[p]
    y = np.dot(w, x)

    # regle d'apprentissage hebbien
    hebb_rule = eta*y*x
    oja_rule = eta*y*(x-y*w)
    w = w + oja_rule

    print('Iteration ', i)
    plt.subplot(212); #plt.cla()
    plt.scatter( x1, x2, c='b' )  
    plt.plot([0, w[0]], [0, w[1]], '-r')
    plt.axhline(0, ls=':', color='k')                    
    plt.axvline(0, ls=':', color='k')                    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis('scaled')                                    
    plt.draw()

    plt.pause(0.1)

print('La norme du vecteur de poids : %.1f' % (np.sqrt(np.sum(w**2))) )

plt.subplot(211)
plt.arrow(0, 0, w[0], w[1], color='g', lw=2)
plt.draw()


plt.show()