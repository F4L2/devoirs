import numpy as np

from junk import *

dic1 = init_dic("reseau1.txt")
dic2 = init_dic("reseau2.txt")
dic3 = init_dic("reseau3.txt")

deg1 = deg(dic1)
deg2 = deg(dic2)
deg3 = deg(dic3)


# plus le degré est fort plus le noeud est ancien

#réseau scale free = réseau à taille exponentielle. 
#réseau 1 et 3 scale free car les degrés des noeuds varient beaucoup.