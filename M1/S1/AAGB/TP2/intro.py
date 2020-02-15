import numpy as np
import sys
import itertools as it

#from junk import *


def ultrametric (matrice):
        print (matrice)

        #doit être carré et contenir au moins 3 espèces
        if(matrice.shape[0] != matrice.shape[1] or matrice.shape[0] < 3):
                return False

        
        # mat = formatage(matrice)
        # #print(mat)

        # comb = np.array(list(it.combinations(mat, 3)))
        # triplet = []

        # for c in comb:
        #         tri = list(it.product(*c))
        #         triplet.append( tri )
        #         # print(tri)
        # triplet = np.array( triplet )

        # for c in triplet:
        #         for t in c:
        #                 print(t)
        #                 if( t[0] <= max(t[1], t[2]) ):
        #                         continue
        #                 return False
        

        index = np.arange(len(matrice))
        comb = np.array(list(it.combinations(index, 3)))

        for c in comb:
                for x in range(len(matrice[c[0]])):
                        if(x == c[0]):
                                continue
                        for y in range(len(matrice[c[1]])):
                                if(y == c[1] or (y == c[0] and x == c[1]) ):
                                        continue
                                for z in range(len(matrice[c[2]])):
                                        if(z == c[2] or (z == c[0] and x == c[2]) or (z == c[1] and y == c[2]) ):
                                                continue

                                        print( matrice[c[0]][x], matrice[c[1]][y], matrice[c[2]][z])
                                        if( matrice[c[0]][x] <= max(matrice[c[1]][y], matrice[c[2]][z]) ):
                                                continue
                                        return False
        return True



def additive (matrice):
        #doit être carré et contenir au moins 4 espèces
        if(matrice.shape[0] != matrice.shape[1] or matrice.shape[0] < 4):
                return False

        #compléter

        return True