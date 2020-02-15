# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

# environnement
LX = 1000        # mm

# cellule de lieu
A = 100.         # Hz, activite maximale
S = 50.          # mm, largeur du champs recepteur
x_pref = 400.    # mm, coordonnee x de la position preferee

# tableaux pour sauvegarder l'activit� de la cellule de lieu et 
# la position correspondante
r = []
pos = []

vit = 160       # mm/s, vitesse du rat
dt = 0.1        # s, temps entre des enregistrements succ�sifs 

# position initiale du rat dans l'environnement
x = 0

while x<LX:
    
    # activite de la cellule de lieu dans la position actuelle
    act = A*np.exp( -(x_pref-x)**2/(2*S**2))
    r.append( act )
    pos.append(x)
    
    # le rat se d�place vers la droite
    x = x + vit*dt

# Figure 
plt.figure(1); plt.show(); plt.clf()
 
plt.subplot(211)
plt.plot(pos, r)
plt.xlim(0,LX)
plt.ylim(0,150)
plt.title('Tuning curve d`une cellule de lieu en 1D')
plt.xlabel('Position du rat dans le couloir')
plt.ylabel('Activite la cellule de lieu , Hz')
plt.draw()


