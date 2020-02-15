# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

# Temps
T   = 50.                  # ms, Temps total de la simulation
dt  = 0.1                   # ms, Pas de temps de la simulation
t   = np.arange(0,T,dt)     # ms, tableau des temps en lesquels sont calcul�es les variables
N   = len(t)                # Longueur du tableau des temps 

# Stimulation par activit� al�atoitre de M neurones pre-synaptiques
i_amp = 0.006     # microA, amplitude de courant inject� par un PA

# entrees
image = np.load('predator.npy')
M = image.shape[0]*image.shape[1]    # M neurones presynaptiques
entrees = image.reshape((M,))        # chaque neurone voit un pixel

# Excitabilit�
tau_m = 10.     # ms, Constante de temps membranaire
R_m   = 4000.   # KOhm, R�sistance de membrane 
theta = 20.     # mV, Seuil du potentiel d'action
v_P   = 100.    # mV, Potentiel au pic du PA 

# Initialisation
i_inj  = np.zeros((M,N))
v      = np.zeros((M,N))

# Simulation
tps_pa = np.zeros(M)    # temps de 1er PA

for i in range(1,N):

    # entrees
    i_inj[:,i]  = i_amp*entrees
            
    # potentiel membranaire
    v[:,i] = v[:,i-1] + dt/tau_m * ( -v[:,i-1] + R_m * i_inj[:,i-1]  )

    # detection de PA
    if any(v[:,i-1] > theta):
        v[v[:,i-1] > theta,i] = v_P
        
    # remise a� 0
    if any(v[:,i-1] == v_P):
        v[v[:,i-1] == v_P,i] = 0;


# decoder l'image � partir du temps du 1er PA
delta=np.zeros(M)
for i in range(M):
    indice_pa = np.where(v[i,:]>theta)[0]
    if len(indice_pa>0):
        delta[i] = indice_pa[0]*dt
    else:
        delta[i] = np.max(delta)

image_decode = 1-delta/np.max(delta)


# Figures
plt.figure(1);  plt.clf();

# Figure image present� 
plt.subplot(311)
plt.imshow(image, cmap='gray')

# 1er PA d'un sousensemble de 50 neurones 
plt.subplot(312)
sousensemble = np.random.choice(range(M), 50, replace=False)
for i,n in enumerate(sousensemble):
    indice_pa = np.where(v[n,:]>theta)[0]
    if len(indice_pa)>0:
        plt.scatter(indice_pa[0]*dt, i+1, s=1, color='k', marker='.');
plt.ylim(0, 51)
plt.xlim(0, T)
plt.xlabel('Temps (ms)')
plt.ylabel('# neurone')
plt.title('PA presynaptiques')
plt.draw()

# Figure potentiel du neurone post-synaptique
plt.subplot(313)
plt.imshow(np.reshape(image_decode, image.shape), cmap='gray' )
plt.draw()


plt.show();