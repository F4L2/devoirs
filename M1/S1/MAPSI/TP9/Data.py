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

def tirage( m ):
    return np.random.uniform(-m, m), np.random.uniform(-m ,m)

def monteCarlo( N ):
    abs = []
    ord = []
    pi = 0
    for i in range(N):
        x,y = tirage(1)
        if( np.sqrt(x**2 + y**2) <= 1):
            pi += 1
        abs.append(x)
        ord.append(y)

    return (pi/N)*4, np.array(abs), np.array(ord)


def swapF( Taut , c1, c2):
    Tau = Taut.copy()

    Tau[c1] = c2
    Tau[c2] = c1

    return Tau

def decrypt ( mess, tau):
    mes = ""
    for c in mess:
        mes += tau[c]
    return mes

def logLikelihood (mess, mu , A, chars2index):
    ml = np.log(mu[chars2index[mess[0]]])
    prev = mess[0]
    for c in range(1, len(mess)):
        ml += np.log( A[ chars2index[prev] ][ chars2index[mess[c]] ] )
        prev = mess[c]
    return ml

def MetropolisHastings(mess, mu, A, tau, N, chars2index):
    mess_p = decrypt( mess, tau_p )
    ml = logLikelihood (mess_p, mu , A, chars2index)

    for i in range[N]:
        c1 , c2 = random.choice(list(tau.items()))
        tau_p = swapF(tau, c1, c2)
        mess_p = decrypt( mess, tau_p )
        ml = logLikelihood (mess_p, mu , A, chars2index)


###############################

# plt.figure()

# # trace le carré
# plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')

# # trace le cercle
# x = np.linspace(-1, 1, 100)
# y = np.sqrt(1- x*x)
# plt.plot(x, y, 'b')
# plt.plot(x, -y, 'b')

# # estimation par Monte Carlo
# pi, x, y = monteCarlo(int(1e4))

# # trace les points dans le cercle et hors du cercle
# dist = x*x + y*y
# plt.plot(x[dist <=1], y[dist <=1], "go")
# plt.plot(x[dist>1], y[dist>1], "ro")
# plt.show()

# print(pi)

# si vos fichiers sont dans un repertoire "ressources"
with open("countWar.pkl", 'rb') as f:
    (count, mu, A) = pkl.load(f, encoding='latin1')

with open("secret.txt", 'r') as f:
    secret = f.read()[0:-1] # -1 pour supprimer le saut de ligne
 
tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }

# n_tau = swapF(tau)
# print(n_tau)

tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }
print(decrypt ( "aabcd", tau ))
print(decrypt ( "dcba", tau ))

# chars2index = dict(zip(np.array(list(count.keys())), np.arange(len(count.keys()))))
with open("fichierHash.pkl", 'rb') as f:
    chars2index = pkl.load(f, encoding='latin1')

print(logLikelihood( "abcd", mu, A, chars2index ))
print(logLikelihood( "dcba", mu, A, chars2index ))



'''
f= open('data.pkl','wb')
pkl.dump(data,f) # penser à sauver les données pour éviter de refaire les opérations
f.close()
'''