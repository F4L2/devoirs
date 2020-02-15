import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5,step=20,data=None):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :return: une matrice 2d contenant les points de la grille, la liste x, la liste y
    """
    if data is not None:
        xmax,xmin,ymax,ymin = np.max(data[:,0]),np.min(data[:,0]),\
                              np.max(data[:,1]),np.min(data[:,1])
    x,y = np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step),
                      np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)


def norme(x):
    #distance vers (0,0)
    #return np.sqrt(x[0]**2 + x[1]**2)
    return np.linalg.norm(x)

def log_norme(x, x_opt):
    return np.array( [np.log(norme(xt-x_opt)) for xt in x] ) 

def optimize(fonc,dfonc,xinit,eps,max_iter=300):
    x_histo = []
    f_histo = []
    grad_histo = []

    x= xinit
    x_opt = x
    last_norm = norme(x)
    for i in range(max_iter):
        x_histo.append(x) 
        f = fonc(x)
        f_histo.append(f)
        grad = dfonc(x)
        grad_histo.append(grad)

        x = x - eps*grad
        if(norme(x_opt) > norme(x)):
            x_opt = x

        # print(abs(last_norm - norme(x)))
        if(abs(last_norm - norme(x)) <= eps):
            break
        
        eps /= np.sqrt(1+i) 

    x_histo = np.array(x_histo)
    f_histo = np.array(f_histo)
    grad_histo = np.array(grad_histo)
    return x_histo, f_histo, grad_histo, x_opt


def f(x):
    return x * np.cos(x)
def df(x):
    return np.cos(x)-x*np.sin(x)

def g(x):
    return -np.log(x) + x**2
def dg(x):
    return -1/x + 2*x


def h(x):
    return np.array(100 * (x[:,1] - x[:,0]**2)**2 + (1-x[:,0])**2)
def dh(x):
    x1 = x[:,0]
    x2 = x[:,1]
    dx1 = np.array(-200*x2+400*x1+2*x1+2)
    dx2 = np.array(200*x2-200*x1)
    return np.c_[dx1.ravel(),dx2.ravel()]

#je ne prend qu'un point pendant l'optimisation
def func_h(x):
    x1 = x[0]
    x2 = x[1]
    return 100 * (x2 - x1**2)**2 + (1-x1)**2
def grad_h(x):
    x1 = x[0]
    x2 = x[1]
    dx1 = np.array(-200*x2+400*x1+2*x1+2)
    dx2 = np.array(200*x2-200*x1)
    return np.array([dx1,dx2])
################################################################################################

#(7291,256)
train = load_usps("USPS/USPS_train.txt")
#(2007,256)
test = load_usps("USPS/USPS_test.txt")

'''afficher les fonctions'''
n= 30
grid,xx,yy = make_grid(-3,3,-3,3,n)

'''dim1'''
view = np.linspace(0,n)

# mafonction = f
# mongrad = df
# plt.figure()
# plt.plot(mafonction(view), label="f")
# plt.plot(mongrad(view),  label="df")
# plt.legend()
# plt.savefig("f")

# mafonction = g
# mongrad = dg
# plt.figure()
# plt.plot(mafonction(view), label="g")
# plt.plot(mongrad(view),  label="dg")
# plt.legend()
# plt.savefig("g")

'''dim2'''
# mafonction = h
# plt.figure()
# plt.contourf(xx,yy,mafonction(grid).reshape(xx.shape))
# plt.savefig("h_2D")

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(xx, yy, mafonction(grid).reshape(xx.shape),rstride=1,cstride=1,\
# 	  cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
# fig.colorbar(surf)
# plt.savefig("h_3D")

'''optimise'''
eps = 0.001
x0 = 1 #point initiale random

xt, fxt, dfxt, x_opt = optimize(f, df, x0, eps)
plt.figure()
plt.plot(f(view))
plt.plot(f(xt))
plt.gca().legend( ('f', 'f_optimizing') )
plt.savefig("optimizing_f")

xt, fxt, dfxt, x_opt = optimize(g, dg, x0, eps)
plt.figure()
plt.plot(g(view))
plt.plot(g(xt))
plt.gca().legend( ('g', 'g_optimizing') )
plt.savefig("optimizing_g")

x0 = [1,1] #point random 2D
xt, fxt, dfxt, x_opt = optimize(func_h, grad_h, x0, eps)
plt.figure()
plt.contourf(xx,yy,h(grid).reshape(xx.shape))
plt.scatter(xt[:,0], xt[:,1])
plt.savefig("optimizing_h")


#log norme
#1D
view = np.linspace(0,n)
plt.figure()
plt.plot( log_norme(xt,x_opt) )
plt.savefig("log1D")

#2D
mafonction = log_norme
plt.figure()
plt.contourf(xx,yy,mafonction(grid, x_opt).reshape(xx.shape))
plt.savefig("log2D")

#3D
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, mafonction(grid, x_opt).reshape(xx.shape),rstride=1,cstride=1,\
	  cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
fig.colorbar(surf)
plt.savefig("log3D")

''' régression logistique optionelle '''
