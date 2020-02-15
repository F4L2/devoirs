import numpy as np
import sys

from junk import *



def ultrametric (matrice):
        #doit être carré et contenir au moins 3 espèces
        if(matrice.shape[0] != matrice.shape[1] or matrice.shape[0] < 3):
                return False

        mat = formatage(matrice)

        comb = np.array(list(it.combinations(mat, 3)))
        triplet = []

        for c in comb:
                tri = list(it.product(*c))
                triplet.append( tri )
                # print(tri)
        triplet = np.array( triplet )

        for c in triplet:
                for t in c:
                        print(t)
                        if( t[0] <= max(t[1], t[2]) ):
                                continue
                        return False
                        
        return True



def additive (matrice):
        #doit être carré et contenir au moins 4 espèces
        if(matrice.shape[0] != matrice.shape[1] or matrice.shape[0] < 4):
                return False

        mat = formatage(matrice)

        return True