'''
N° étudiant: 3870665
'''
import numpy as np


def mse_batch(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    """pour plot_error"""
    f_x = datax.dot(w)
    output = ((datay[:] - f_x[:])**2).sum()
    return output/len(datax)

def mse_g_batch(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    f_x = datax.dot(w)
    output = (-2*datax[:] * (datay[:] - f_x[:])).sum()
    return output/len(datax)

def hinge_batch(datax,datay,w):
    """ retourne la moyenne de l'erreur hinge """
    """pour plot_error"""
    f_x = datax[:].dot(w)
    output = (np.maximum(0, -datay[:]*f_x[:])).sum()
    return output/len(datax)

def hinge_g_batch(datax,datay,w , marge = 0):
    """ retourne le gradient moyen de l'erreur hinge """
    output = np.where( datay*datax.dot(w) > marge, 0, marge -datay * datax)
    return output.sum()/len(datax)




def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    """version stochastique"""
    f_x = datax.dot(w)
    return (datay - f_x)**2

def mse_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    """version stochastique"""
    f_x = datax.dot(w)
    return -2*datax * (datay - f_x)

def hinge(datax,datay,w, marge = 0):
    """ retourne la moyenne de l'erreur hinge """
    """version stochastique"""
    
    f_x = datax.dot(w)
    return np.maximum(0, marge -datay*f_x)

def hinge_g(datax,datay,w , marge = 0):
    """ retourne le gradient moyen de l'erreur hinge """
    """version stochastique"""
    if(datay * datax.dot(w) > marge ):
        return 0
    else:   
        return marge -datay * datax #transposer datax

