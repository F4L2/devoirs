# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

# le nombre d'epreuves
K = 1000

# le nombre d'actions (direction d'une saccade)
N = 4

# recompenses moyennes (le nombre de gouttes de jus)
E_r = np.array([ 2., 4., 6., 8.])

# parametres du modele
eta = 0.1
eps = 0.14

# initialisation de la fonction valeur
Q = np.zeros( N )

mem_Q  = np.zeros((K,N));
mem_erreur = np.zeros(K)

# recompense totale obtenue
R = 0

# boucle pour K epreuves
for i in range(K):

    # action aleatoire, exploration
    # remplacer par la strategie epsilon-glouton
    if( np.random.random() > eps ):
        a = np.argmax(Q)   
    else:
        a = np.random.randint(N) 

    # recompense aleatoire avec la moyenne E_r[a]
    r =  np.random.poisson(E_r[a])

    # mettre a jour l'estimation de la fonction valeur
    Q[a] += eta * (r - mem_Q[i-1][a]) #/!\ 1er iter prend la dernière val de la mémoire

    # memoriser l'estimation de Q pour visualisation
    mem_Q[i] = Q

    if a == 2:   # saccade vers une cible
        # memoriser l'activite dopaminergique associee
        mem_erreur[i] = r - Q[a]

    # recompense totale
    R = R + r;

print('Recompense totale obtenue : %d' % R)

# initialisation de graphisme
plt.figure(1); plt.clf()

# distribution de recompenses
r0 = np.random.poisson(E_r[0], 1000)
r1 = np.random.poisson(E_r[1], 1000)
r2 = np.random.poisson(E_r[2], 1000)
r3 = np.random.poisson(E_r[3], 1000)

plt.subplot(421)
plt.hist(r0, color = 'b', label = 'r1')
plt.axvline(E_r[0], lw=2, color='k')
plt.legend()
plt.xlabel('recompense')
plt.ylabel('# de fois')
plt.draw()

plt.subplot(422)
plt.hist(r1, color = 'g', label = 'r2')
plt.axvline(E_r[1], lw=2, color='k')
plt.legend()
plt.draw()

plt.subplot(423)
plt.hist(r2, color = 'r', label = 'r3')
plt.axvline(E_r[2], lw=2, color='k')
plt.legend()
plt.draw()

plt.subplot(424)
plt.hist(r3, color = 'c', label = 'r4')
plt.axvline(E_r[3], lw=2, color='k')
plt.legend()
plt.draw()


# estimation de la fonction valeur
plt.subplot(413)
plt.plot( mem_Q[:,0], 'b-', lw=2)
plt.plot( mem_Q[:,1], 'g-', lw=2)
plt.plot( mem_Q[:,2], 'r-', lw=2)
plt.plot( mem_Q[:,3], 'c-', lw=2)
plt.axhline(E_r[0], ls = ':', lw=2, color='b')
plt.axhline(E_r[1], ls = ':', lw=2, color='g')
plt.axhline(E_r[2], ls = ':', lw=2, color='r')
plt.axhline(E_r[3], ls = ':', lw=2, color='c')
plt.ylim(-1,10)
plt.xlabel('epreuves')
plt.ylabel('Q(a)')
plt.draw()

# erreur de prevision de recompense
plt.subplot(414)
plt.stem( range(len(mem_erreur)), mem_erreur)
plt.xlabel('epreuves quand a=2 est choisie')
plt.axhline(0, ls=':', color='k')
plt.ylabel('erreur de prevision')
plt.draw()

plt.show()
