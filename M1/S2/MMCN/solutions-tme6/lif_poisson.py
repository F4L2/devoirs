# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

# Temps
T   = 500.                  # ms, Temps total de la simulation
dt  = 0.1                   # ms, Pas de temps de la simulation
t   = np.arange(0,T,dt)     # ms, tableau des temps en lesquels sont calculées les variables
N   = len(t)                # Longueur du tableau des temps 

# Stimulation par activité aléatoitre de M neurones pre-synaptiques
i_amp = 0.1     # microA, amplitude de courant injecté par un PA

M = 10          # M neurones presynaptiques
F = 5           # taux de decharge de neurones presynaptiques
rnd = np.random.rand(M,N);
pa_entree = np.zeros((M,N))
pa_entree[rnd<F/1000.]=1.;

# Excitabilité

tau_m = 10.     # ms, Constante de temps membranaire
R_m   = 4000.   # KOhm, Résistance de membrane 
theta = 20.     # mV, Seuil du potentiel d'action
v_P   = 100.    # mV, Potentiel au pic du PA 

# Initialisation

i_inj  = np.zeros(N)
v      = np.zeros(N)

# Simulation

for i in range(1,N):

    # entrees
    i_inj[i]  = i_amp*np.sum(pa_entree[:,i])
            
    # potentiel membranaire
    v[i] = v[i-1] + dt/tau_m * ( -v[i-1] + R_m * i_inj[i]  )

    # detection de PA
    if v[i-1] > theta:
        v[i] = v_P

    # remise ˆ 0
    if v[i-1] == v_P:
        v[i] = 0;


plt.figure(1); plt.show(); plt.clf();

# Figure PA presynaptiques
plt.subplot(211)
for i in range(M):
    indice_pa = np.where(pa_entree[i,:])[0]
    plt.scatter(indice_pa,(i+1)*np.ones(len(indice_pa)), s=1, color='k', marker='.');
plt.ylim(0, M+1)
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
