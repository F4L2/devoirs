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


def bernoulli( p ):
    test = np.random.random_sample()
    if test > p:
        return False
    return True

def binomiale(n,p):
    res = 0
    for i in range (n):
        if bernoulli(p):
            res += 1
    return res

def galton(n,p):

    v = np.arange(0, n)
    for i in range(n):
        v[i] = binomiale(n,p)

    t = np.zeros(n)
    dif = 0
    for i in range(n):
        t[v[i]] += 1
    for i in range(n):
        if t[i] != 0:
            dif+=1

    plt.hist(v, bins = dif, color = 'blue', edgecolor = 'red')
    plt.ylabel('nb_apparition')
    plt.xlabel('valeur')
    plt.title('historigramme')
    plt.show()



def normale ( k, sigma ):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    
    
    y = sigma * np.random.randn(k) #randn(k) donne k résultat de N(0,1), pas intéressant

    print(y)

    plt.figure()
    plt.plot(y)
    plt.show()

    return y
    

def proba_affine ( k, slope ):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    if abs ( slope  ) > 2. / ( k * k ):
        raise ValueError ( 'la pente est trop raide : pente max = ' +
        str ( 2. / ( k * k ) ) )
    
    y = np.arange(0, k)

    if(slope != 0):
        for i in range (k):
            y[i] = (1/k) - ((i - ( (k - 1) / 2)) * slope)
    else : 
        for i in range (k):
            y[i] = 1/k
    
    plt.figure()
    plt.plot(y)
    plt.show()

    return y



#galton(1000,0.5)
#normale(11, 1)
#proba_affine( 121 , 1/5553333)









#Indépendance conditionnelle

def independant( A, B ,R ):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if(A[i,j,k].all() * B[i,j,k].all() != R[i,j,k].all()):
                    return False
    return True

def independant_2( A, B ,R ):
    for x in range(2):
        for y in range(2):
            for z in range(2):
                if(A[x].all() * B[x,y].all() != R[x,y,z].all()):
                    return False
    return True


# creation de P(X,Y,Z,T)
P_XYZT = np.array([[[[ 0.0192,  0.1728],
                     [ 0.0384,  0.0096]],

                    [[ 0.0768,  0.0512],
                     [ 0.016 ,  0.016 ]]],

                   [[[ 0.0144,  0.1296],
                     [ 0.0288,  0.0072]],

                    [[ 0.2016,  0.1344],
                     [ 0.042 ,  0.042 ]]]])

P_YZ = np.zeros((2,2))
P_YZ[0][0] = P_XYZT[:, 0, 0, :].sum()    # X
P_YZ[0][1] = P_XYZT[:, 0, 1, :].sum()    # Y
P_YZ[1][0] = P_XYZT[:, 1, 0, :].sum()    # Z
P_YZ[1][1] = P_XYZT[:, 1, 1, :].sum()    # T


P_XTcondYZ = np.zeros((2,2,2,2)) #????
for x in range(2):
    for y in range(2):
        for z in range(2):
            for t in range(2):
                P_XTcondYZ[x,y,z,t] = P_XYZT[x,y,z,t] / P_YZ[y,z]

#print(P_XTcondYZ)


P_XcondYZ = np.zeros((2,2,2))
P_TcondYZ = np.zeros((2,2,2))

for x in range(2):
    for y in range(2):
        for z in range(2):
            P_XcondYZ[x,y,z] = P_XTcondYZ[x,y,z,:].sum()

for y in range(2):
    for z in range(2):
        for t in range(2):
            P_TcondYZ[y,z,t] = P_XTcondYZ[:,y,z,t].sum()



#print( independant(P_TcondYZ, P_XcondYZ, P_XTcondYZ) )

P_XYZ = np.zeros((2,2,2))
for x in range(2):
    for y in range(2):
        for z in range(2):
            P_XYZ[x,y,z] = P_XYZT[x,y,z,:].sum()

P_X = np.zeros((2))
P_X[0] = P_XYZ[0,:,:].sum()
P_X[1] = P_XYZ[1,:,:].sum()

P_XY = np.zeros((2,2))
for x in range(2):
    for y in range(2):
        P_XY[x,y] = P_XYZ[x,y,:].sum()

#print( independant_2(P_X, P_XY, P_XYZ))



'''
f= open('coordVelib.pkl','wb')
pkl.dump(data,f) # penser à sauver les données pour éviter de refaire les opérations
f.close()
'''