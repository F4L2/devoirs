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


# old version = python 2
#data = pkl.load(file("ressources/lettres.pkl","rb"))
# new :
with open('lettres.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1')
X = np.array(data.get('letters')) # récupération des données sur les lettres
Y = np.array(data.get('labels')) # récupération des étiquettes associées 


# affichage d'une lettre
def tracerLettre(let):
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]])
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("exlettre.png")
    return

def discretise(X , d):
    etats = d
    intervalle = 360 / etats
    Y = []
    for x in X:
        Y.append(np.floor(x/intervalle))
    return Y


def groupByLabel(y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index


def learnMarkovModel(Xc, d):
    A = np.zeros((d,d))
    Pi = np.zeros(d)

    for c in Xc:
        Pi[int(c[0])] += 1

    for c in range (len(Xc)):
        ignore = True
        for x in Xc[c]:
            if(ignore == True):
                ignore = False
                mem = x
                continue
            A[int(mem)][int(x)] += 1
            mem = x

    A = A/np.maximum(A.sum(1).reshape(d,1),1) # normalisation 
    if(Pi.sum() != 0): 
        Pi = Pi/Pi.sum()
    
    return [Pi, A]


def probaSequence(s,Pi,A):
    prob = np.log(Pi[int(s[0])])

    for i in range( len(s) - 1 ):
        prob += np.log( A[int(s[i])][int(s[i+1])] ) 

    return prob

# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:int(np.floor(pc*n))])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

########################
'''
d = 3
Xd = discretise(X, d)
C = groupByLabel(Y)

Xc = []
for i in range (len(C[0])):
    Xc.append(Xd[i]) 

piA = learnMarkovModel(Xc, d)
print(piA)
'''

d=3     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe
models = []
inc = 0
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    Xc = []
    for i in range( inc, len(index[cl]) + inc ):
        Xc.append(Xd[i])
        
    inc += len (index[cl])
    #print(inc)
    models.append(learnMarkovModel(Xc, d))

res = []
for m in models:
    res.append( probaSequence( Xc[0] ,m[0], m[1]) )
    #print(m)

print(res)

proba = np.array([[probaSequence(Xd[i], models[cl][0], models[cl][1]) for i in range(len(Xd))]for cl in range(len(np.unique(Y)))])

#calcul d'une version numérique des Y 
Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num

#Calcul de la classe la plus probable
pred = proba.argmax(0) # max colonne par colonne

#Calcul d'un pourcentage de bonne classification 
print(np.where(pred != Ynum, 0.,1.).mean())

''' On testait sur les données en apprentissage. On va séparer les données en entrée en 2 jeu de donnée, train et test. '''
''' On aura forcément bon sur les données que l'on a appris, inutile de tester dessus. '''
''' il faut refaire le test en tenant compte de la séparation '''

# exemple d'utilisation
itrain,itest = separeTrainTest(Y,0.8)

ia = []
for i in itrain:
    ia += i.tolist()    
it = []
for i in itest:
    it += i.tolist()

#print(ia)
#print(it)


'''
f= open('data.pkl','wb')
pkl.dump(data,f) # penser à sauver les données pour éviter de refaire les opérations
f.close()
'''
