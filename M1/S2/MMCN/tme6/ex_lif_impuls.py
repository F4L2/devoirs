# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

# Temps

T   = 500.                  # ms, Temps total de la simulation
dt  = 0.1                   # ms, Pas de temps de la simulation
t   = np.arange(0,T,dt)     # ms, tableau des temps en lesquels sont calcul�es les variables
N   = len(t)                # Longueur du tableau des temps 

# Stimulation par un des PA entrants

i_amp = 0.1                 # microA, amplitude de courant inject� par un PA
pa = np.zeros(N)            # le tableau des temps des PA entrants

t_pa = [int(i/dt) for i in range(100,200,2)]  	    # ms, le temps d'un PA entrant

pa[ t_pa ] = 1.     	    # le PA entrant
#il faut qu'il y ait assez de PA entrant arrivant avec une fréquence assez forte pour provoquer un PA sortant(solution de v) avec l'accumulation des pulses. 

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

    # courant inject�
    if pa[i]>0:
        i_inj[i-1]  = i_amp
            
    # potentiel membranaire
    v[i] = v[i-1] + dt/tau_m * ( -v[i-1] + R_m * i_inj[i-1]  )

    # detection de PA
    if v[i-1] > theta:
        v[i] = v_P

    # remise � 0
    if v[i-1] == v_P:
        v[i] = 0;


plt.figure(1); plt.clf();

# Figure Courant
plt.subplot(211)
plt.plot(t,i_inj);
plt.ylim(0, 0.5)
plt.xlabel('Temps (ms)')
plt.ylabel('I, uA')
plt.title('PA entrant')
plt.draw()


# Figure potentiel du neurone post-synaptique
plt.subplot(212)
plt.plot(t,v)
plt.ylim(0, 30)
plt.xlabel('Temps (ms)')
plt.ylabel('V (mV)')
plt.title('Potentiel membranaire post-synaptique')
plt.draw()

plt.show();