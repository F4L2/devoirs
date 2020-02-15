'''
N° étudiant: 3870665
'''
import numpy as np
from arftools import * 

#####Given
def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.colorbar()

def plot_error(datax,datay,f,name,step=10):
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.savefig(name)
    plt.show()
########

def split_data(setx, sety, percent):
    data = list(zip(setx,sety))
    np.random.shuffle(data)
    train_tresh = int(sety.size*percent)
    X,Y = zip(*data)
    X = np.array(X)
    Y = np.array(Y)
    return X[:train_tresh], Y[:train_tresh], X[train_tresh:], Y[train_tresh:]

def class_versus_class(setx, sety, a, b):
    datax = []
    datay = []
    for x,y in zip(setx,sety):
        if(y==a):
            datay.append(1)
            datax.append(x)
        elif(y==b):
            datay.append(-1)
            datax.append(x)
    return np.array(datax), np.array(datay)

def class_versus_all(setx, sety, a):
    datax = []
    datay = []
    for x,y in zip(setx,sety):
        if(y==a):
            datay.append(1)
            datax.append(x)
        else:
            datay.append(-1)
            datax.append(x)
    return np.array(datax), np.array(datay)

