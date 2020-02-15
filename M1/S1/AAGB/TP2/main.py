'''
N° étudiant: 3870665
'''

import numpy as np
import sys

from upgma import *
from nj import *
from intro import *


'''

Introduction:

1.  Une matrice de distance est additive si elle respecte la condition des 4 points pour tous ses quartets.
    La condition des 4 points pour le quartet i,j,k,l est
    satisfaite si deux des sommes sont les mêmes, avec la
    troisième somme plus petite que les premières deux.

    Une matrice de distance est ultramétrique si:  dist(x,y) <= max{ dist(x,z), dist(y,z) } 

TODO: finir les tests sur matrice. 

'''



M1 = np.array([   [spanish_inquisition,8,7,12], 
                  [8,spanish_inquisition,9,14], 
                  [7,9,spanish_inquisition,11], 
                  [12,14,11,spanish_inquisition] ])

M2 = np.array([   [spanish_inquisition,2,3,8,14,18],
                  [2,spanish_inquisition,3,8,14,18],
                  [3,3,spanish_inquisition,8,14,18],
                  [8,8,8,spanish_inquisition,14,18],
                  [14,14,14,14,spanish_inquisition,18],
                  [18,18,18,18,18,spanish_inquisition] ])
#UPGMA
M3 = np.array([   [spanish_inquisition,19,27,8,33,18,13],
                  [19,spanish_inquisition,31,18,36,1,13],
                  [27,31,spanish_inquisition,26,41,32,29],
                  [8,18,26,spanish_inquisition,31,17,14],
                  [33,36,41,31,spanish_inquisition,35,28],
                  [18,1,32,17,35,spanish_inquisition,12],
                  [13,13,29,14,28,12,spanish_inquisition] ])
#Neighbor Joining
M4 = np.array([   [spanish_inquisition,2,4,6,6,8],
                  [2,spanish_inquisition,4,6,6,8],
                  [4,4,spanish_inquisition,6,6,8],
                  [6,6,6,spanish_inquisition,4,8],
                  [6,6,6,4,spanish_inquisition,8],
                  [8,8,8,8,8,spanish_inquisition] ])




#valeur inconnue/doublon = maxsize pour ne pas gêner durant la recherche du min
#je met spanish_inquisition pour être lisible mais devrait être sys.maxsize
#le mieux serais d'avoir le tableau complet 
matU = np.array([       [spanish_inquisition, spanish_inquisition, spanish_inquisition, spanish_inquisition, spanish_inquisition, spanish_inquisition, spanish_inquisition],
                        [19, spanish_inquisition, spanish_inquisition, spanish_inquisition, spanish_inquisition, spanish_inquisition, spanish_inquisition],
                        [27, 31, spanish_inquisition, spanish_inquisition, spanish_inquisition, spanish_inquisition, spanish_inquisition],
                        [8, 18, 26, spanish_inquisition, spanish_inquisition, spanish_inquisition, spanish_inquisition],
                        [33, 36, 41, 31, spanish_inquisition, spanish_inquisition, spanish_inquisition], 
                        [18, 1, 32, 17, 35, spanish_inquisition, spanish_inquisition],
                        [13, 13, 29, 14, 28, 12, spanish_inquisition]
                ])

matN = np.array([ [spanish_inquisition,2,4,6,6,8],
                  [2,spanish_inquisition,4,6,6,8],
                  [4,4,spanish_inquisition,6,6,8],
                  [6,6,6,spanish_inquisition,4,8],
                  [6,6,6,4,spanish_inquisition,8],
                  [8,8,8,8,8,spanish_inquisition]
                ])
########################

# tests sur matrices (NOT WORKING) 
# print( ultrametric(M2) )

# construction d'arbre (WORKING)
newick_U = upgma( matU ) 
newick_N = nj( M4 )

print("EXEMPLE UPGMA: \t", newick_U)
print("EXEMPLE NJ: \t", newick_N)