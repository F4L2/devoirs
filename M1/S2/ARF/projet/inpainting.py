from image import *

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


def find_dic_weight(training_dic, patch, reg):
    training_dic = training_dic.T
    patch = patch.T
    reg.fit(training_dic , patch)  
    dic_weight = reg.coef_

    return dic_weight


def priority( pix_coord, center ):
    c_x, c_y = center
    distance_coord = []
    for x,y in pix_coord:
        distance_coord.append( np.linalg.norm( x-c_x + y-c_y ) )

    priority_liste = sorted( zip( distance_coord, pix_coord) )
    priority_liste = [ tpl[1] for tpl in priority_liste ]
    priority_liste = priority_liste[::-1] #sort from most distant to closest

    return priority_liste


