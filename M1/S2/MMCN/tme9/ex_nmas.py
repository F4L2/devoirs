# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

# le nombre d'epreuves
K = 5000

# le nombre d'etats 
# le nombre de machines a sous, chacune correspond a une couleur du croix de fixation
S = 3

# le nombre d'actions (direction de saccade) dans chaque etat 
N = 4

# recompenses moyennes aleatoires (le nombre de gouttes de jus)
E_r = np.random.randint(0,20, (S,N))

# parametres du modele
eta = 0.05  #si ça 'grésille', c'est trop grand
eps = 0.1

# initialisation
Q = np.zeros( (S,N) )

mem_Q  = np.zeros((K,S,N))

# boucle pour K epreuves
for i in range(K):

    # choix aleatoire d'une machine a sous (ou d'un etat)
    s = np.random.randint(S)

    # choix d'une action, strategie epsilon-greedy
    if( np.random.random() > eps ):
        a = np.argmax(Q[s])   
    else:
        a = np.random.randint(N) 

    # recompense
    r =  np.random.poisson(E_r[s,a])

    # mettre a jour l'estimation de la fonction valeur
    Q[s,a] += eta * (r - mem_Q[i-1][s,a]) 

    # garder les resultats pour visualisation
    mem_Q[i]= Q


# initialisation de graphisme
plt.figure(1); plt.clf()

# estimation de la fonction valeur

mem_Q = np.array(mem_Q)

Q_mas0 = mem_Q[:, 0, :]
Q_mas1 = mem_Q[:, 1, :]
Q_mas2 = mem_Q[:, 2, :]

plt.subplot(311)
plt.plot(Q_mas0 )
plt.axhline(E_r[0,0], ls = ':', color='k')
plt.axhline(E_r[0,1], ls = ':', color='k')
plt.axhline(E_r[0,2], ls = ':', color='k')
plt.axhline(E_r[0,3], ls = ':', color='k')
plt.ylim(-1)
plt.ylabel('Q(a)')
plt.title('Machine a sous 1')
plt.draw()

plt.subplot(312)
plt.plot(Q_mas1 )
plt.axhline(E_r[1,0], ls = ':', color='k')
plt.axhline(E_r[1,1], ls = ':', color='k')
plt.axhline(E_r[1,2], ls = ':', color='k')
plt.axhline(E_r[1,3], ls = ':', color='k')
plt.ylim(-1)
plt.ylabel('Q(a)')
plt.title('Machine a sous 2')
plt.draw()


plt.subplot(313)
plt.plot(Q_mas2 )
plt.axhline(E_r[2,0], ls = ':', color='k')
plt.axhline(E_r[2,1], ls = ':', color='k')
plt.axhline(E_r[2,2], ls = ':', color='k')
plt.axhline(E_r[2,3], ls = ':', color='k')
plt.ylim(-1)
plt.xlabel('epreuves')
plt.ylabel('Q(a)')
plt.title('Machine a sous 3')
plt.draw()

plt.show()
