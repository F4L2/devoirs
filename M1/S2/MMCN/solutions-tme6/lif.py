# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

# Temps

T   = 500.                  # ms, Temps total de la simulation
dt  = 0.1                    # ms, Pas de temps de la simulation
t   = np.arange(0,T,dt)     # ms, tableau des temps en lesquels sont calculées les variables
N   = len(t)                # Longueur du tableau des temps 

# Stimulation par un créneau de courant injecté

i_amp = 0.006   # uA, Courant injecté pendant le créneau
t_on  = 100.    # ms, Début du créneau
t_off = 400.    # ms, Fin du créneau

# Excitabilité

tau_m = 10.     # ms, Constante de temps membranaire
R_m   = 4000.   # KOhm, Résistance de membrane 
theta = 20.     # mV, Seuil du potentiel d'action
v_P   = 100.    # mV, Potentiel au pic du PA 

# Initialisation

i_inj  = np.zeros(N)
v      = np.zeros(N)

# Simulation
num_pa = 0;
tps_pa = [];

for i in range(1,N):

    # courant injecté
    if (t[i]>=t_on) and (t[i]<t_off):
        i_inj[i]  = i_amp
            
    # potentiel membranaire
    v[i] = v[i-1] + dt/tau_m * ( -v[i-1] + R_m * i_inj[i-1]  )

    # detection de PA
    if v[i-1] > theta:
        v[i] = v_P
        tps_pa.append(i*dt)

    # remise ˆ 0
    if v[i-1] == v_P:
        v[i] = 0
        num_pa += 1


print('Temps du 1er PA')
print('Simulation: %.2f ms' % (tps_pa[0]-t_on))
print('Theorique: %.2f ms' % (tau_m*np.log(R_m*i_amp/(R_m*i_amp-theta)))  )
print('')
print('Frequence')
print('Simulation: %.2f Hz' % (num_pa*1000./(t_off-t_on)) )
print('Theorique: %.2f Hz' % (1000./(tau_m*np.log(R_m*i_amp/(R_m*i_amp-theta)))) )
 
plt.figure(1); plt.show(); plt.clf();

# Figure Courant
plt.subplot(211)
plt.plot(t,i_inj);
plt.ylim(0, 0.010)
plt.xlabel('Temps (ms)')
plt.ylabel('I, uA')
plt.title('Courant injecte')
plt.draw()


# Figure potentiel du neurone post-synaptique
plt.subplot(212)
plt.plot(t,v)
plt.ylim(0, 110)
plt.xlabel('Temps (ms)')
plt.ylabel('V (mV)')
plt.title('Potentiel membranaire post-synaptique')
plt.draw()
