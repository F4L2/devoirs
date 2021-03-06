# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

N = 20          # nombre de cellules

# environnement
LX = 1000        # mm

# cellules de lieu
A = 100.         # Hz, activite maximale
S = 50.          # mm, largeur du champs recepteur
x_pref = np.linspace(0,LX,N)    # mm, positions preferees de N cellules

# tableaux pour sauvegarder l'activit� de la cellule de lieu et 
# la position correspondante
r = []
pos = []

# position initiale du rat dans l'environnement
vit = 160       # mm/s, vitesse du rat
dt = 0.1        # s, temps entre des enregistrements succ�sifs 

x = 0


# Figure 
plt.figure(1); plt.show(); plt.clf()

erreur = []
erreur2 = []

while x<LX:
    
    # activite de la cellule de lieu dans la position actuelle
    act = A*np.exp( -(x_pref-x)**2/(2*S**2))
    r.append( act )
    pos.append(x)
    
    x_est  = x_pref[np.argmax(act)]
    x_est2 = sum(act*x_pref)/sum(act)

    erreur.append(np.abs(x-x_est))
    erreur2.append(np.abs(x-x_est2))

    plt.subplot(211); plt.cla()
    plt.plot(x_pref, act, 'o')
    plt.axvline(x, ls=':')
    plt.axvline(x_est, color='r', lw=2)
    plt.axvline(x_est2, color='g', lw=2)
    plt.ylim(0,150)
    plt.xlim(0,LX)
    plt.xlabel('Position du rat dans le couloir')
    plt.ylabel('Activite, Hz')
    plt.draw()

    plt.pause(0.1)

    # le rat se d�place vers la droite
    x = x + vit*dt
    
 
plt.subplot(212)
plt.plot(pos, r)
plt.ylim(0,150)
plt.title('Tuning curves des cellules de lieu en 1D')
plt.xlabel('Position du rat dans le couloir')
plt.ylabel('Activite la cellule de lieu , Hz')
plt.draw()

plt.figure(2); plt.clf()
plt.plot(erreur, 'r-', label='argmax')
plt.plot(erreur2, 'g-', label = 'pop.  vector')
plt.ylabel('Erreur')
plt.legend()
plt.draw()
plt.show()

