'''
N° étudiant: 3870665
'''
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')   
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb

import time
import pickle as pkl


# truc pour un affichage plus convivial des matrices numpy
np.set_printoptions(precision=2, linewidth=320)
plt.close('all')

data = pkl.load(file("ressources/lettres.pkl","rb"))
X = np.array(data.get('letters'))
Y = np.array(data.get('labels'))

nCl = 26



'''
f= open('data.pkl','wb')
pkl.dump(data,f) # penser à sauver les données pour éviter de refaire les opérations
f.close()
'''
