import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from sklearn.neighbors import KNeighborsClassifier

plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    ## extent pour controler l'echelle du plan

poidata = pickle.load(open("data/poi-paris.pkl","rb"))
## liste des types de point of interest (poi)
print("Liste des types de POI" , ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "night_club"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]),2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]


# ## Affichage brut des poi
# show_map()
# ## alpha permet de regler la transparence, s la taille
# plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)
# plt.savefig(typepoi+"_POI")


###################################################

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 100

x_cases = np.linspace(xmin,xmax,steps)
y_cases = np.linspace(ymin,ymax,steps)

xx,yy = np.meshgrid(x_cases,y_cases)
grid = np.c_[xx.ravel(),yy.ravel()]

# # A remplacer par res = monModele.predict(grid).reshape(steps,steps)
# res = np.random.random((steps,steps))
# plt.figure()
# show_map()
# plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
#                alpha=0.3,origin = "lower")
# plt.colorbar()
# plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.3)


###################################
'''             TP              '''
#python3 -i tme2-poi.py

#geo_mat[i][1] : xi
#geo_mat[i][0] : yi

#grid[i][0] : xi
#grid[i][1] : yi


# #histogramme
# H = np.zeros( (len(xx), len(yy)) )
# for i in range(len(geo_mat)):
#     xi = geo_mat[i][1]
#     yi = geo_mat[i][0]
#     ids_xy = [0,0]
#     for ix in range(len(x_cases)-1):
#         if(xi >= x_cases[ix] and xi <= x_cases[ix+1]):
#             ids_xy[0] = ix
#             break
#         for iy in range(len(y_cases)-1):
#             if(yi >= y_cases[iy] and yi <= y_cases[iy+1]):
#                 ids_xy[1] = iy
#                 break
#     H[ids_xy[0]][ids_xy[1]] += 1

# # flat_H = np.array( [item for sublist in H for item in sublist] )

# # x, y: inversé sur matplotlib
# res = H
# plt.figure()
# show_map()
# plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
#                alpha=0.3,origin = "lower")
# plt.colorbar()
# plt.savefig(typepoi+"_histogramme"+str(steps))



#Parzen
#phi= fonction de la fenêtre 
#1: fonction indicatrice, dit si le point est dedans ou pas. 

# P = np.zeros( (steps, steps) )
# h= 0.0001
# dim = 2
# vol = h**dim
# phi = 1/(vol) # fois val
# for i in range(len(grid)):
#     x0 = grid[i][0]
#     y0 = grid[i][1]
#     val = 0
#     #chaque point à affecter
#     for j in range(len(geo_mat)):
#         xi = geo_mat[j][1]
#         yi = geo_mat[j][0]

#         #si pas dans la fenêtre pas besoin de chercher plus loin. val = 0 ==> phi*val=0
#         #fonction indicatrice
#         if( abs(xi-x0) >= h/2 or abs(yi-y0) >= h/2):
#             continue
#         #la fonction indicatrice est validée, val = 1
#         val += phi     
#     index_x = list(x_cases).index(x0)
#     index_y = list(y_cases).index(y0)
#     P[index_x][index_y] += (val / len(geo_mat))

# # x, y: inversé sur matplotlib
# res = P
# plt.figure()
# show_map()
# plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
#                alpha=0.3,origin = "lower")
# plt.colorbar()
# plt.savefig(typepoi+"_parzen"+str(steps)+"hpetit")


# pour l'histogramme plus la discrétisation est forte (plus de case), l'estimation de densité est plus précise et on peut mieux distinguer les régions d'intérêts 
# du bruit.  
# plus gros grains
# plus h est petit plus ça ressemble à l'histogramme

# les paramètres servent à régler la taille des cases pour Parzen, ce qui influe aussi l'estimation de densité.
# on peut aussi voir qu'ils règlent le sur apprentissage, si l'on prenait un h vraiment petit, on serait en train d'estimer tous les points un par un. 

# les paramètres sont forcément adaptés au modèle, donc on choisis les paramètres qui permettent la meilleure évaluation du modèle.
# Dans le cas parzen, ce serait tester plein de h (sur un interval, ou lorsque le modèle devient moins bon ou ne change pas beaucoup) et choisir le h qui permet 
# la meilleure estimation de densité














#tous les POI pour KNN
population = [ ]
x_base = []
y_base = []
for typ in poidata.keys():
    ind = list(poidata.keys()).index(typ)
    p_coor = np.zeros((len(poidata[typ]),2))
    for i,(k,v) in enumerate(poidata[typ].items()):
        p_coor[i,:]=v[0]
        poi = list(np.repeat(ind, len(p_coor)))

    x_base.extend(p_coor)
    y_base.extend(poi)
    population.append( [p_coor, poi] )

x_base = np.array(x_base)
y_base = np.array(y_base)
# print(x_base)
# print(x_base.shape)
# print(y_base.shape)

classe = population[5]
target = 0 #furniture store
n = 5 
dist_max = 65.6
neigh = KNeighborsClassifier(n_neighbors= n)
neigh.fit(x_base, y_base) 
# print(neigh.predict([x_base[0]]))
print(x_base[0])
print(neigh.predict([x_base[0]]))
print(neigh.predict_proba([x_base[0]]))
print(neigh.score(classe[0], classe[1], sample_weight=None))

# print(neigh.kneighbors([x_base[0]]))


KNN = np.zeros( (steps, steps) )
for x in grid:
    x0 = x[0]
    y0 = x[1]
    x = [x[1], x[0]]
    
    # dist, point = neigh.kneighbors([x])
    # dist = dist[0]
    # point = point[0]

    val = 1000
    if(neigh.predict([x])[0] != target):
        val = 0
    val *= neigh.predict_proba([x])[0][target]

    # #sup ou inf, on supprime
    # for j in range(n):
    #     if(dist[j] > dist_max):
    #         dist[j]= 0
    #     elif(dist[j] <  -dist_max):
    #         dist[j]= 0
    # #print(dist)
    
    # #mauvais pred , ignorer
    # for j in range(n):
    #     if(y_base[point[j]] == neigh.predict([x])):
    #         point[j]= 1
    #     else:
    #         point[j]= 0

    index_x = list(x_cases).index(x0)
    index_y = list(y_cases).index(y0)
    # KNN[index_x][index_y] += sum( dist[:]*point[:] )
    # KNN[index_x][index_y] += sum( dist[:] )
    KNN[index_x][index_y] += val

print(KNN)

# x, y: inversé sur matplotlib
plt.figure()
show_map()
plt.imshow(KNN,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.savefig("KNN")
