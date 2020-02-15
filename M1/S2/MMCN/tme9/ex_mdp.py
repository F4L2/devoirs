# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# le nombre d'epreuves
K = 500

# le nombre d'etats 
# le nombre de machines a sous, chacune correspond a une position dans le couloir
S = 11
T1 = 0       # l'�tat terminal a gauche
T2 = S-1     # l'�tat terminal a droite

# le nombre d'actions (directions de mouvement) dans chaque etat 
#   a=0 aller a droite
#   a=1 aller a gauche
N = 2

# parametres du modele
eta = 0.1
eps = 0.1
gam = 0.9

# initialisation de la fonction valeur
Q = np.zeros((S,N))            

mem_Q  = np.zeros((K,S,N))

# initialisation de graphisme
plt.figure(1); plt.clf()

# boucle pour K epreuves
for i in range(K):

    # mettre l'agent a l'etat initial
    s = 2

    # choisir la 1ere action, strategie epsilon-greedy
    if np.random.rand() < eps :
        a = np.random.randint(N)    # action aleatoire, exploration
    else:
        a = np.argmax(Q[s,:])       # action optimale, exploitation

    # repeter jusqu'a la fin de l'epreuve
    while s != T1 and s != T2:     # l'�tat terminal

        # effectuer l'action choisie et passer a l'etat s`
        if a==0:    
            if s<T2: s_new = s + 1       # aller a droite, mais pas depasser le mur
            else: s_new = s

        elif a==1:       
            if s>T1: s_new = s - 1       # aller a gauche, mais pas depasser le mur
            else: s_new = s

        else:
            print("ERREUR: cette action n'existe pas !")

        # obtenir la recompense si dans l'�tat terminal
        if s_new == T2: 
            r = 1.
        else:
            r = 0.

        # choisir la nouvelle action a` dans l'etat s`
        if np.random.rand() < eps :
            a_new = np.random.randint(N)    # action aleatoire, exploration
        else:
            a_new = np.argmax(Q[s_new,:])   # action optimale, exploitation

        # mettre a jour l'estimation de la fonction valeur 
        delta = r+gam*Q[s_new,a_new]-Q[s,a]
        Q[s,a] += eta*delta

        # mettre a jour l'etat et l'action
        a = a_new
        s = s_new


    # visualisation de l'apprentissage de la fonction erreur
    if i%10==0:
        print('Iteration %d' % i)
        ax = plt.subplot(222, projection='3d')
        X,Y = np.meshgrid(range(S), range(N))
        ax.plot_wireframe(X,Y,Q.T)
        plt.xlabel('etats')
        plt.ylabel('actions')
        plt.yticks([0,1])
        plt.title('Q(s,a)')
        plt.ylim(-0.2,1.2)
        plt.pause(0.01)

    # garder les resultats pour visualisation
    mem_Q[i] = Q




# valeur de Q(s,a=0) pendant l'apprentissage pour s=2, s=5 et s=8
Q_s2_a0 = mem_Q[:,2,0]
Q_s5_a0 = mem_Q[:,5,0]
Q_s8_a0 = mem_Q[:,8,0]

# valeurs de Q(s,a) apres l'apprentissage pour tous les etats
Q_s_a0 = mem_Q[K-1,:,0]
Q_s_a1 = mem_Q[K-1,:,1]

# valeurs theorique pour les deux actions
Q_theor0 = gam**np.array([8,7,6,5,4,3,2,1,0])
Q_theor1 = np.insert(gam**np.array([9,8,7,6,5,4,3,2]), 0, 0)


plt.subplot(221)
plt.plot(Q_s2_a0, label = 's=2, a=0', color='b' )
plt.plot(Q_s5_a0, label = 's=5, a=0', color='g' )
plt.plot(Q_s8_a0, label = 's=8, a=0', color='r' )
plt.axhline(gam**7, ls = ':', lw=2, color='b')
plt.axhline(gam**4, ls = ':', lw=2, color='g')
plt.axhline(gam**1, ls = ':', lw=2, color='r')
plt.ylim(-0.1, 1.1)
plt.xlabel('epreuves')
plt.ylabel('Q(s,a)')
plt.legend(loc='lower right')
plt.draw()

plt.subplot(223)
plt.plot(Q_s_a0, 'ko-', label = 'appris' )
plt.plot(range(1,T2),Q_theor0, 'ko--', label = 'theorique' )
plt.axhline(1,ls=':',color='k')
plt.xlabel('etats')
plt.ylabel('Q(s,a=0)')
plt.legend(loc='upper left')
plt.ylim(0,1.5)
plt.draw()

plt.subplot(224)
plt.plot(Q_s_a1, 'ko-', label = 'appris' )
plt.plot(range(1,T2), Q_theor1, 'ko--', label = 'theorique' )
plt.axhline(1,ls=':',color='k')
plt.xlabel('etats')
plt.ylabel('Q(s,a=1)')
plt.legend(loc='upper left')
plt.ylim(0,1.5)
plt.draw()

plt.show() 

