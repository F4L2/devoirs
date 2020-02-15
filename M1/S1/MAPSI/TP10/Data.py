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
import numpy.random as rd



##############################################################
a = 6.
b = -1.
N = 100
sig = .4 # écart type
x = rd.rand(1,N)
y = a*x + b + rd.rand(1, N) * sig

#print(x)
plt.figure()
plt.scatter(x, y)

a_est = (np.cov(x[0],y[0]) / np.var(x[0]))[1][0]
b_est = (y[0].mean() - ( np.cov(x[0],y[0]) / np.var(x[0]) ) * x[0].mean())[0][1]

y_curve1 = a_est *x[0] + b_est
plt.plot(x[0], y_curve1)


X = np.hstack((x.reshape(N,1),np.ones((N,1))))

A = np.dot(np.transpose(X), X)
B = np.dot(np.transpose(X), y[0])

y_curve2 = np.linalg.solve(A,B)
y_curve2 = x[0] * y_curve2[0] + x[0] * y_curve2[1] 
plt.plot(x[0], y_curve2)





wstar = np.linalg.solve(X.T.dot(X), X.T.dot(y[0])) # pour se rappeler du w optimal

eps = 5e-4
nIterations = 400
w = np.zeros(X.shape[1]) # init à 0
allw = [w]
for i in range(nIterations):
    # A COMPLETER => calcul du gradient vu en TD
    w = [ w[0] - eps * 2 * sum( - x[0] * ( y[0] - w[0] * x[0] - w[1] ) ) , w[1] - eps * sum( 2 * -1 * ( y[0] - w[0] * x[0] - w[1]) ) ]
    
    allw.append(w)

allw = np.array(allw)

#print(allw)

# tracer de l'espace des couts
ngrid = 20
w1range = np.linspace(-0.5, 8, ngrid)
w2range = np.linspace(-1.5, 1.5, ngrid)
w1,w2 = np.meshgrid(w1range,w2range)

cost = np.array([[np.log(((X.dot(np.array([w1i,w2j]))-y)**2).sum()) for w1i in w1range] for w2j in w2range])

plt.figure()
plt.contour(w1, w2, cost)
plt.scatter(wstar[0], wstar[1],c='r')
plt.plot(allw[:,0],allw[:,1],'b+-' ,lw=2 )



from mpl_toolkits.mplot3d import Axes3D

costPath = np.array([np.log(((X.dot(wtmp)-y)**2).sum()) for wtmp in allw])
costOpt  = np.log(((X.dot(wstar)-y)**2).sum())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w1, w2, cost, rstride = 1, cstride=1 )
ax.scatter(wstar[0], wstar[1],costOpt, c='r')
ax.plot(allw[:,0],allw[:,1],costPath, 'b+-' ,lw=2 )


c = 1
plt.figure()
yquad = a * x[0]**2 + b * x[0] + c + eps
plt.plot(x[0], yquad, "+")



data = np.loadtxt("winequality/winequality-red.csv", delimiter=";", skiprows=1)
N,d = data.shape # extraction des dimensions
pcTrain  = 0.7 # 70% des données en apprentissage
allindex = np.random.permutation(N)
indTrain = allindex[:int(pcTrain*N)]
indTest = allindex[int(pcTrain*N):]
X = data[indTrain,:-1] # pas la dernière colonne (= note à prédire)
Y = data[indTrain,-1]  # dernière colonne (= note à prédire)
# Echantillon de test (pour la validation des résultats)
XT = data[indTest,:-1] # pas la dernière colonne (= note à prédire)
YT = data[indTest,-1]  # dernière colonne (= note à prédire)

plt.figure()
ywine = a * X[0]**2 + b * X[0] + c + eps
plt.plot(X[0], ywine, "+")

plt.show()

'''
f= open('data.pkl','wb')
pkl.dump(data,f) # penser à sauver les données pour éviter de refaire les opérations
f.close()
'''
