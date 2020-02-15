# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

# Temps
T   = 500.                  # ms, Temps total de la simulation
dt  = 0.1                   # ms, Pas de temps de la simulation
t   = np.arange(0,T,dt)     # ms, tableau des temps en lesquels sont calcul�es les variables
N   = len(t)                # Longueur du tableau des temps 

# Stimulation par activit� al�atoitre de M neurones pre-synaptiques
i_amp = 0.1     # microA, amplitude de courant inject� par un PA

F = 10                      # taux de decharge de neurone presynaptiques
rnd = np.random.rand(N);
pa_entree = np.zeros(N)
pa_entree[rnd<F/1000.]=1;   # cr�er un train al�atoire de PA avec frequence F

# Excitabilit�

tau_m = 10.     # ms, Constante de temps membranaire
R_m   = 4000.   # KOhm, R�sistance de membrane 
theta = 20.     # mV, Seuil du potentiel d'action
v_P   = 100.    # mV, Potentiel au pic du PA 

# Initialisation

i_inj  = np.zeros(N)
v      = np.zeros(N)

# Simulation

for i in range(1,N):

    # entrees
    i_inj[i]  = i_amp*pa_entree[i]
            
    # potentiel membranaire
    v[i] = v[i-1] + dt/tau_m * ( -v[i-1] + R_m * i_inj[i]  )

    # detection de PA
    if v[i-1] > theta:
        v[i] = v_P

    # remise � 0
    if v[i-1] == v_P:
        v[i] = 0;


plt.figure(1); plt.clf();

# Figure PA presynaptiques
plt.subplot(211)
indice_pa = np.where(pa_entree)[0]
plt.scatter(indice_pa,np.ones(len(indice_pa)), s=1, color='k', marker='.');
plt.ylim(0, 2)
plt.xlabel('Temps (ms)')
plt.ylabel('# neurone')
plt.title('PA presynaptiques')
plt.draw()


# Figure potentiel du neurone post-synaptique
plt.subplot(212)
plt.plot(t,v)
plt.ylim(0, 110)
plt.xlabel('Temps (ms)')
plt.ylabel('V (mV)')
plt.title('Potentiel membranaire post-synaptique')
plt.draw()


plt.show();


#ajouter des neuronnes pour provoquer des PA sortants