import numpy as numpy
from arftools import *


class Loss ( object ) :
    def __init__(self, cost, grad):
        self.cost = cost
        self.grad = grad

    def forward ( self , y , yhat ) :
        #calcul le cout
        return self.cost(y,yhat)
    
    def backward ( self , y, yhat ) :
        #calcul le gradient du cout
        return self.grad(y,yhat) #comment calculer sans x ??



def init_w(X):
    if(len(X[0].shape)>0):
        size = np.size(X,1)
    else:
        size = np.size(X)
    W = np.zeros(size)
    W += np.random.rand(size)
    W /= W.sum()
    return W

class Module ( object ) :
    def __init__(self, activ, eps = 0.01):
        self._parameters = None
        self._gradient = None
        self.eps = eps
        self.activ_func = activ

    def init_w(self, X):
        if(len(X[0].shape)>0):
            size = np.size(X,1)
        else:
            size = np.size(X)
        W = np.zeros(size)
        W += np.random.rand(size)
        W /= W.sum()
        self._parameters = W

    def zero_grad (self) :
        ## Annule  gradient
        self._gradient = 0

    def update_parameters ( self , gradient_step ) :
        ## Calcule la mise a jour des parametres selon le gradient et le pas de gradient_step
        self._parameters -= gradient_step * self._gradient
        

    def forward (self, X) :
        ## Calcule la passe forward     
        self.init_w(X)
        
        return self.activ_func(X, self._parameters)

    def backward_delta ( self , input ,  delta ) :
        ## Calcul la derivee de l’erreur

        #sum_k ( delta[k] ) * derive(forward(input) par rapport à input) 
        return sum(delta[:] * self._parameters[:])
        
    def backward_update_gradient ( self , input ,  delta ) :
        ## Met a jour la valeur du gradient

        grad = delta * input
        self._gradient += self.eps * grad
        pass



'''func = activation'''
def lineaire(z, W):
    return z.dot(W) 
def lineaire_g(z, W):
    return z 

def tanh(z, ignored=None):
    return (1- np.exp(-2*z)) / (1 + np.exp(-2*z))
def tanh_g(z, ignored=None):
    return 1-tanh(z)**2

def sigmoid(z, ignored=None):
    return (1 / (1 + np.exp(-z)))
def sigmoid_g(z, ignored=None):
    return sigmoid(z) * (1-sigmoid(z))

'''func = cout'''
def mse(y,yhat):
    return sum( (y - yhat)**2 )

def mse_g(x, y, yhat):
    return 2*x*sum( (y - yhat) )




#xor 4 points
base = [  np.array( [ [0,0], [0,1], [1,0], [1,1] ]) ,  np.array( [-1, 1, 1, -1])  ]
# print(base[0].shape)

M1 = Module(lineaire)
ychap = M1.forward(base[0])
print(ychap)
M2 = Module(sigmoid)
ychap = M2.forward(ychap)
print(ychap)

loss = Loss(mse, mse_g)
cout = loss.forward(base[1],ychap)
print(cout)

grad = loss.backward(base[1], ychap)
print(grad)



# # #(7291,256)(7291,)
# train = load_usps("USPS/USPS_train.txt")
# print(train[1].shape)
# # #(2007,256)(2007,)
# test = load_usps("USPS/USPS_test.txt")