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
    return np.sqrt(x[0]**2 + x[1]**2)


def optimize(fonc,dfonc,xinit,eps,max_iter):
    x_histo = []
    f_histo = []
    grad_histo = []

    x= xinit
    # x_opt = x
    for i in range(max_iter):
        
        x_histo.append(x) 
        f = fonc(x)
        f_histo.append(f)
        grad = dfonc(x)
        grad_histo.append(grad)

        # print(x)
        x = x - eps*grad
        # if(norme(x_opt) > norme(x)):
        #     x_opt = x

    x_histo = np.array(x_histo)
    f_histo = np.array(f_histo)
    grad_histo = np.array(grad_histo)
    return x_histo, f_histo, grad_histo


def f(x):
    return x * np.cos(x)
def df(x):
    return 1*np.cos(x) + x*(- np.sin(x))

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

grid,xx,yy = make_grid(-1,3,-1,3,20)

'''dim1'''
# view = np.linspace(0,20)
# mafonction = f
# plt.figure()
# plt.plot(mafonction(view))
# plt.show()

'''dim2'''
# mafonction = h
# plt.figure()
# plt.contourf(xx,yy,mafonction(grid).reshape(xx.shape))

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(xx, yy, mafonction(grid).reshape(xx.shape),rstride=1,cstride=1,\
# 	  cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
# fig.colorbar(surf)
# plt.show()


'''optimise'''
n= 200
eps = 0.0001
x0 = np.array([3,5])

xt, fxt, dfxt = optimize(func_h, grad_h, x0, eps, n) 
#faire un plus plus précis.... 
x_opt = xt[-1] 


# print(xt.shape)
# print(fxt.shape)
# print(dfxt.shape)

#ne marche pas en 2D
# plt.figure()
# plt.plot(xt, label='xt')
# plt.plot(fxt, label='f(xt)')
# plt.plot(dfxt, label='grad(f(xt))')
# plt.gca().legend( ('xt', 'f_xt', 'grad_xt') )
# plt.show()





def log_norme(x, x_opt):
    return np.array( [np.log(norme(xt-x_opt)) for xt in x] ) 

view = np.linspace(0,n)
plt.figure()
plt.plot( log_norme(xt,x_opt) )
plt.savefig("log1D")


mafonction = log_norme
plt.figure()
plt.contourf(xx,yy,mafonction(grid, x_opt).reshape(xx.shape))
plt.savefig("log2D")

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, mafonction(grid, x_opt).reshape(xx.shape),rstride=1,cstride=1,\
 	  cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
fig.colorbar(surf)
plt.savefig("log3D")
plt.show()
