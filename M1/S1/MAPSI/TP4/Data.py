'''
N° étudiant: 3870665
'''
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')   
import matplotlib.animation as animation

import time
import pickle as pkl

from math import *
from pylab import *

def read_file ( filename ):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break

    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )
    infile.close ()

    # transformation de la liste en tableau 2D
    data = np.asarray ( data )
    data.shape = ( int ( data.size / 2 ), 2 )

    return data

def dessine_1_normale ( params ):
    # récupération des paramètres
    mu_x, mu_z, sigma_x, sigma_z, rho = params

    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    # création de la grille
    x = np.linspace ( x_min, x_max, 100 )
    z = np.linspace ( z_min, z_max, 100 )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm = X.copy ()
    for i in range ( x.shape[0] ):
        for j in range ( z.shape[0] ):
            norm[i,j] = normale_bidim ( x[i], z[j], params )

    # affichage
    fig = plt.figure ()
    plt.contour ( X, Z, norm, cmap=cm.autumn )
    plt.show ()

def dessine_normales ( data, params, weights, bounds, ax ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]

    # création de la grille
    nb_x = nb_z = 100
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm0 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], params[0] )# * weights[0]
    norm1 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
             norm1[j,i] = normale_bidim ( x[i], z[j], params[1] )# * weights[1]

    # affichages des normales et des points du dataset
    ax.contour ( X, Z, norm0, cmap=cm.winter, alpha = 0.5 )
    ax.contour ( X, Z, norm1, cmap=cm.autumn, alpha = 0.5 )
    for point in data:
        ax.plot ( point[0], point[1], 'k+' )


def find_bounds ( data, params ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # calcul des coins
    x_min = min ( mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:,0].min() )
    x_max = max ( mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:,0].max() )
    z_min = min ( mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:,1].min() )
    z_max = max ( mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:,1].max() )

    return ( x_min, x_max, z_min, z_max )



def learn_parameters ( data ): #unused
    output = np.zeros((5)) #0: x mu, 1: z mu, 2: x sigma, 3: z sigma, 4: cov(x,z)

    output[0] = data[:,0].mean() 
    output[1] = data[:,1].mean()
    output[2] = data[:,0].std() 
    output[3] = data[:,1].std()
    output[4] = np.corrcoef(data[:,0], data[:,1]) [1,0]

    return output

def normale_bidim( x, z, param):
    n = 1/(2 * np.pi * param[2] * param[3] * np.sqrt(1 - pow(param[4], 2)))    
    crochet = pow( ((x - param[0]) / param[2]) , 2) - (2* param[4]) * ( ((x - param[0]) * (z - param[1]))  / (param[2] * param[3]) ) + pow( ((z - param[1]) / param[3]), 2 ) 
    e = np.exp( -(1/ (2*(1 - pow(param[4], 2) ))) * crochet)

    return n * e


def affiche (data):
    # affichage des données : calcul des moyennes et variances des 2 colonnes
    mean1 = data[:,0].mean ()
    mean2 = data[:,1].mean ()
    std1  = data[:,0].std ()
    std2  = data[:,1].std ()

    # les paramètres des 2 normales sont autour de ces moyennes
    params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                        (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
    weights = np.array ( [0.4, 0.6] )
    bounds = find_bounds ( data, params )

    # affichage de la figure
    fig = plt.figure ()
    ax = fig.add_subplot(111)
    dessine_normales ( data, params, weights, bounds, ax )
    plt.show ()

def Q_i (data, c_params, c_weights):
     output = np.zeros( (len(data), 2) )
     
     for i in range( len(data) ):
         alpha0 = normale_bidim(data[i,0], data[i,1], c_params[0]) * c_weights[0]
         alpha1 = normale_bidim(data[i,0], data[i,1], c_params[1]) * c_weights[1]
         output[i, 0] = alpha0 / (alpha0 + alpha1)
         output[i, 1] = alpha1 / (alpha0 + alpha1)

     return output

def  M_step (data, Q, c_params, c_weights):
    sum_Q0 = Q[:,0].sum()
    sum_Q1 = Q[:,1].sum()

    new_weights = np.zeros( (2) )
    new_params = np.zeros( (2,5) )

    new_weights[0] = sum_Q0 / (sum_Q0 + sum_Q1)
    new_weights[1] = sum_Q1 / (sum_Q0 + sum_Q1)

    new_params[0,0] = (Q[:,0] * data[:,0]).sum() / sum_Q0
    new_params[0,1] = (Q[:,0] * data[:,1]).sum() / sum_Q0
    new_params[0,2] = np.sqrt( (Q[:,0] * (data[:,0] - new_params[0,0])**2 ).sum() / sum_Q0)
    new_params[0,3] = np.sqrt( (Q[:,0] * (data[:,1] - new_params[0,1])**2 ).sum() / sum_Q0)
    new_params[0,4] = (Q[:,0] * ( ((data[:,0] - new_params[0,0])*(data[:,1] - new_params[0,1])) / (new_params[0,3] * new_params[0,2])) ).sum() / sum_Q0


    new_params[1,0] = (Q[:,1] * data[:,0]).sum() / sum_Q1
    new_params[1,1] = (Q[:,1] * data[:,1]).sum() / sum_Q1
    new_params[1,2] = np.sqrt( (Q[:,1] * (data[:,0] - new_params[1,0])**2 ).sum() / sum_Q1) 
    new_params[1,3] = np.sqrt( (Q[:,1] * (data[:,1] - new_params[1,1])**2 ).sum() / sum_Q1)
    new_params[1,4] = (Q[:,1] * ( ((data[:,0] - new_params[1,0])*(data[:,1] - new_params[1,1])) / (new_params[1,3] * new_params[1,2])) ).sum() / sum_Q1

    return new_params, new_weights

def EM (data, params, weights):

    for i in range(4):
        bounds = find_bounds ( data, params )
        
        fig = plt.figure ()
        ax = fig.add_subplot(111)
        dessine_normales ( data, params, weights, bounds, ax )
        plt.show ()

        Q = Q_i ( data, params, weights )
        params, weights = M_step ( data, Q, params, weights ) 
        

def EM_2 (data, params, weights):
    output = [ [params, weights] ]

    for i in range(20):
        Q = Q_i ( data, params, weights )
        params, weights = M_step ( data, Q, params, weights ) 

        iter = [params, weights]
        output.append(iter)

    return output 

# calcul des bornes pour contenir toutes les lois normales calculées
def find_video_bounds ( data, res_EM ):
    bounds = np.asarray ( find_bounds ( data, res_EM[0][0] ) )
    for param in res_EM:
        new_bound = find_bounds ( data, param[0] )
        for i in [0,2]:
            bounds[i] = min ( bounds[i], new_bound[i] )
        for i in [1,3]:
            bounds[i] = max ( bounds[i], new_bound[i] )
    return bounds


# la fonction appelée à chaque pas de temps pour créer l'animation
def animate ( i ):
    ax.cla ()
    dessine_normales (data, res_EM[i][0], res_EM[i][1], bounds, ax)
    ax.text(5, 40, 'step = ' + str ( i ))
    #print "step animate = %d" % ( i )


# éventuellement, sauver l'animation dans une vidéo
# anim.save('old_faithful.avi', bitrate=4000)

##################################################
data = read_file ( "geyser.txt" )

#res = normale_bidim ( 1, 2, (1.0,2.0,3.0,4.0,0) )
#print(res)

#dessine_1_normale ( (-3.0,-5.0,3.0,2.0,0.7) )
#dessine_1_normale ( (-3.0,-5.0,3.0,2.0,0.2) )

#affiche(data)

current_params = np.array([[ 3.28778309, 69.89705882, 1.13927121, 13.56996002, 0. ],
                           [ 3.68778309, 71.89705882, 1.13927121, 13.56996002, 0. ]])

# current_weights = np.array ( [ pi_0, pi_1 ] )
current_weights = np.array ( [ 0.5, 0.5 ] )

#T = Q_i ( data, current_params, current_weights )
#print(T)


current_params = np.array([[ 3.2194684, 67.83748075, 1.16527301, 13.9245876,  0.9070348 ],
                           [ 3.75499261, 73.9440348, 1.04650191, 12.48307362, 0.88083712]])
current_weights = np.array ( [ 0.49896815, 0.50103185] )
#T = Q_i ( data, current_params, current_weights )
#print(T)

current_params = array([(2.51460515, 60.12832316, 0.90428702, 11.66108819, 0.86533355),
                        (4.2893485,  79.76680985, 0.52047055,  7.04450242, 0.58358284)])
current_weights = array([ 0.45165145,  0.54834855])

Q = Q_i ( data, current_params, current_weights )
new_p, new_w = M_step ( data, Q, current_params, current_weights )
#print(new_p)
#print(new_w)



mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [ 0.5, 0.5 ] )

#EM (data, params, weights)

res_EM = EM_2( data, params, weights )
#print(res_EM)


bounds = find_video_bounds ( data, res_EM )

# création de l'animation : tout d'abord on crée la figure qui sera animée
fig = plt.figure ()
ax = fig.gca (xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))

# exécution de l'animation
anim = animation.FuncAnimation(fig, animate, frames = len ( res_EM ), interval=500 )
plt.show ()


'''
f= open('data.pkl','wb')
pkl.dump(data,f) # penser à sauver les données pour éviter de refaire les opérations
f.close()
'''
