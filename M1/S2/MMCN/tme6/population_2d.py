# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from time import sleep

# environnement
LX = 1000        # mm
LY = 1000        # mm

# positions preferees des cellules de lieu
N = 30                         # l'effectif de la population de cellules de lieu
A = 100.                        # Hz, activite maximale
S = 100.                        # mm, largeur du champs recepteur
x_pref = np.random.rand(N)*LX   # x preferé
y_pref = np.random.rand(N)*LY   # y preferé

# Figure : distribution des positions preferees dans l'espace
plt.figure(1); plt.show(); plt. clf()
plt.subplot(221)
plt.plot(x_pref, y_pref, 'o')
plt.axis('scaled')
plt.xlim(0,LX)
plt.ylim(0,LY)
plt.title('Positions preferees')


# simulation de mouvements d'un rat et l'enregistrement de l'activite de cellules de lieu
vitesse = 100.      # mm/s, vitesse du rat
dt      = 0.1       # s, pas de temps
T       = 150       # durée de simulation

# initialisation
x       = 500.          # mm, position initiale du rat
y       = 500.          # mm, position initiale du rat
dir     = np.pi/2.      # direction initiale
t       = 0.
        
pop_act = np.zeros(N) 
erreur = np.zeros(T)

for i in range(T) :
        
        # temps
        t = t + dt
        print('Iteration %d de %d' % (i,T))

        # position du rat dans l'environnement
        x = x + dt*vitesse*np.cos(dir)
        y = y + dt*vitesse*np.sin(dir)

        # activite de toutes les cellules de lieu dans la position (x,y)
        for n in range(N):
            pop_act[n] = A*np.exp( -(x_pref[n]-x)**2/(2*S**2) - (y_pref[n]-y)**2/(2*S**2) )

        # estimation de la position
        estx = np.sum(pop_act * x_pref ) / np.sum( pop_act)
        esty = np.sum(pop_act * y_pref ) / np.sum( pop_act)
        erreur[i] = np.sqrt( (estx-x)**2 + (esty-y)**2)

        # sortie de la boucle
        if (x>LX or x<0 or y<0 or y>LY):
            break  

        # choix de direction du mouvement
        dir = dir + np.random.randn()*np.pi/4

        # Figures

        # position de l'animal
        plt.subplot(222); plt.cla()
        plt.plot(x, y, '+')
        plt.plot(estx, esty, 'o')
        plt.axis('scaled')
        plt.xlim(0,LX)
        plt.ylim(0,LY)
        #plt.draw()
        
        [grid_x,grid_y] = np.mgrid[ 0:LX:20, 0:LY:20]
        pref_pos = np.c_[x_pref, y_pref]       
        Z = griddata(pref_pos,pop_act,(grid_x,grid_y), method='linear')

        # activité de la population en 2D
        plt.subplot(223)
        plt.contourf(grid_x,grid_y,Z)
        plt.axis('scaled')
        plt.xlim(0,LX)
        plt.ylim(0,LY)
        #plt.draw()

        # mettre ˆ jour les graphiques
        plt.pause(0.01)

# erreur de l'estimation de la position à partir des cellules de lieu
plt.subplot(224)
plt.plot(erreur)
plt.ylabel('Erreur, mm')
plt.xlabel('Pas de temps')
plt.draw()

