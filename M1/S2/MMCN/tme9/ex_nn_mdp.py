# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import sleep

# le nombre d'epreuves
K = 500

# le nombre d'�tats
S = 11
T1 = 0       # l'�tat terminal a gauche
T2 = S-1     # l'�tat terminal a droite


# le nombre de neurones d'entree (egal au nombre d'etats)
D = S

# le nombre de neurones de sortie (correspond au nombre d'actions)
N = 2

# parametres du modele
eta = 0.1
eps = 0.1
gam = 0.9

# matrice de poids
W = np.zeros((N,D), dtype=float)
# W = np.random.rand(N,D)
# W[:,-1] = [0,0]

Q = np.zeros((S,N))
mem_Q  = []
mem_delta  = []
mem_etats = []

# initialisation de graphisme
plt.figure(1); plt.clf()

# boucle pour K epreuves
for i in range(K):

    print('Epreuve ' + str(i))

    # l'etat initial
    s = 2

    # les activites de neurones d'entree encodent l'etat de l'animal
    x = np.zeros(D)
    x[s] = 1.
    
    # la fonction-valeur est encodee dans l'activite des neurones de sortie
    Q[s,:] = np.dot( W, x)

    # choix d'une action, strategie epsilon-greedy
    if np.random.rand() < eps:
        a = np.random.randint(N)    # action aleatoire, exploration
    else:
        a = np.argmax(Q[s,:])       # action optimale, exploitation

    mem_delta_epreuve = []
    mem_etats_epreuve = []
    # repeter jusqu'a la fin de l'epreuve
    while s!=T1 and s!=T2:

        # effectuer l'action choisie et passer a l'etat s`
        if a==0:    
            if s<T2: s_new = s + 1     # aller a droite, mais pas depasser le mur
            else: s_new = s

        elif a==1:       
            if s>T1: s_new = s - 1       # aller a gauche, mais pas depasser le mur
            else: s_new = s

        else:
            print("ERREUR: cette action n'existe pas !")

        # obtenir la recompense
        if s_new == T2: 
            r = 1.
        else:
            r = 0.

        # l'activite des neurones d'entree dans le nouvel etat s'
        x_new = np.zeros(D)
        x_new[s_new] = 1.

        # les valeurs des actions dans le nouvel etat 
        Q[s_new, :] = np.dot( W, x_new)

        # choisir la nouvelle action a` dans l'etat s`
        if np.random.rand() < eps :
            a_new = np.random.randint(N)    # action aleatoire, exploration
        else:
            a_new = np.argmax(Q[s_new,:])   # action optimale, exploitation

        # mettre a jour la matrice de poids
        delta = r + gam * Q[s_new, a_new] - Q[s,a]  # signal dopaminergique
        W[a, :] = W[a, :]  + eta*delta*x            # plasticite synaptique

        # initialiser la nouvelle epreuve
        a = a_new
        s = s_new
        x = x_new

        # sauvegarder les resultats de l'epreuve pour visualisation
        mem_delta_epreuve.append(delta)  
        mem_etats_epreuve.append(s)  

    if i%100==0:
        #visuaisation des poids
        ax = plt.subplot(221, projection='3d')
        X,Y = np.meshgrid(range(D), range(N))
        ax.plot_wireframe(X,Y,W)
        plt.xlabel("neurones  d'entree")
        plt.ylabel("neurones de sortie")
        plt.yticks([0,1])
        plt.title('Wni')
        #plt.ylim(-0.2,1.2)
        plt.pause(0.01)
        
        #visuaisation des la fonction valeur
        ax = plt.subplot(222, projection='3d')
        X,Y = np.meshgrid(range(S), range(N))
        ax.plot_wireframe(X,Y,Q.T)
        plt.xlabel('etats')
        plt.ylabel('actions')
        plt.yticks([0,1])
        plt.title('Q(s,a)')
        plt.ylim(-0.2,1.2)
        plt.pause(0.01)

    # sauvegarder pour visualisation
    mem_Q.append(Q.copy())  
    mem_delta.append(mem_delta_epreuve)  
    mem_etats.append(mem_etats_epreuve)  


mem_Q = np.array(mem_Q)
mem_delta = np.array(mem_delta)

# valeur de Q(s,a=0) pendant l'apprentissage pour s=2, s=5 et s=8
Q_s2_a0 = mem_Q[:,2,0]
Q_s5_a0 = mem_Q[:,5,0]
Q_s8_a0 = mem_Q[:,8,0]

# valeurs de Q(s,a) apres l'apprentissage pour tous les etats
Q_s_a0 = mem_Q[K-1,:,0]
Q_s_a1 = mem_Q[K-1,:,1]

# valeurs theorique pour les deux axsions
Q_theor0 = gam**np.array([8,7,6,5,4,3,2,1,0])
Q_theor1 = np.insert(gam**np.array([9,8,7,6,5,4,3,2]), 0, 0)

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

input("Paused. Press any key")
print("Continue.")

plt.subplot(212)
for i in range(200):
    plt.cla()
    plt.text(0.1, 0.9, "Erreur delta dans tous les etats")
    plt.text(0.1, 0.8, "Activite d'un neurone dopaminergique")
    plt.text(0.1, 0.7, "Epreuve %d"%i)
    plt.plot(mem_etats[i], mem_delta[i] )
    plt.xlabel('etats')
    plt.ylabel('erreur delta (DA))')
    #plt.legend(loc='upper left')
    plt.xlim(0,11)
    plt.ylim(0,1)
    plt.draw()

    if i<20:
        input()
    else:
        plt.pause(0.1)
    


