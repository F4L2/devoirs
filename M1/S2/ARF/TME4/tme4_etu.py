from arftools import *
from utility import *
from CostFunc import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Lineaire(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g

    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))

        Ws = []
        Cs = []

        for it in range(self.max_iter):
            ind = np.random.randint(N)
            W = self.w.T
            grad = self.loss_g(datax[ind],datay[ind], W) 
            self.w = self.w - self.eps * grad
            Ws.append(self.w[0])

            C = self.loss(datax[:], datay[:], W).sum()
            Cs.append(C/N)

        return np.array(Ws), np.array(Cs)
        
    def fit_batch(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))

        Ws = []
        Cs = []

        for it in range(self.max_iter):
            W = self.w.T
            grad = self.loss_g(datax,datay, W) 
            self.w = self.w - self.eps * grad
            Ws.append(self.w[0])

            C = self.loss(datax, datay, W)

        return np.array(Ws), np.array(Cs)

    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        f_x = []
        for x in datax:
            f_x.append(np.sign( (x[:] * self.w[:]).sum() ))
        return np.array(f_x)

    def score(self,datax,datay):
        score = 0
        N = len(datax)
        
        y_chap = self.predict(datax)
        for i in range(N):
            pred = y_chap[i]
            y = datay[i]
            if( pred == y ):
                score += 1 

        return score / N


def norme(x):
    #distance vers (0,0)
    return np.sqrt( sum(x[:]**2))

def projection_polynomiale(x,xp):
    fen = []
    for x1 in x:
        for x2 in xp:
            fen.append(x1*x2) 
    return np.array(fen)

def noyau_gauss(set1, set2):
    #(fen1.T).dot(fen2) possible, donc noyau admissible
    return np.exp(-np.linalg.norm(set1-set2)**2) #on admet sigma=1


def init_mat(d,k):
    R = []
    for i in range(d):
        R.append(np.random.normal(0, 1, k))
    return np.array(R)


def plot_error_spe(datax,datay,f, Ws,name,step=10):
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.scatter(Ws[:,0],Ws[:,1])
    plt.savefig(name)
    plt.show()



if __name__=="__main__":
    trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    # plt.figure()
    # plot_error(trainx,trainy,mse_batch, "mse_cost")
    # plt.figure()
    # plot_error(trainx,trainy,hinge_batch, "hinge_cost")


    ##################################
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    Ws, Cs = perceptron.fit(trainx,trainy)
    # plot_error_spe(trainx,trainy,hinge_batch, Ws, "W_traj_stoch")
    # plt.figure()    
    # plt.scatter(Ws[:,0],Ws[:,1])
    # plt.savefig("stoch_w")

    perceptron_batch = Lineaire(hinge_batch,hinge_g_batch,max_iter=1000,eps=0.1)
    Ws, Cs = perceptron_batch.fit_batch(trainx,trainy)
    # plot_error_spe(trainx,trainy,hinge_batch, Ws, "W_traj_batch")
    # plt.figure()    
    # plt.scatter(Ws[:,0],Ws[:,1])
    # plt.savefig("batch_w")

    #la trajectoire d'apprentissage est aléatoire en stochastique, et linéaire en batch
    ####################

    #frontières
    # print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    # plt.figure()
    # plot_frontiere(trainx,perceptron.predict,200)
    # plot_data(trainx,trainy)
    # plt.savefig("frontier_stochastic")
    
    # print("Erreur : batch, train %f, test %f"% (perceptron_batch.score(trainx,trainy),perceptron_batch.score(testx,testy)))
    # plt.figure()
    # plot_frontiere(trainx,perceptron_batch.predict,200)
    # plot_data(trainx,trainy)
    # plt.savefig("frontier_batch")   



    ''' Données usps '''
    # #(7291,256)(7291,)
    train = load_usps("USPS/USPS_train.txt")
    # #(2007,256)(2007,)
    test = load_usps("USPS/USPS_test.txt")


    #6vs9
    class69x, class69y = class_versus_class(train[0], train[1], 6, 9)
    train69x, train69y, test69x, test69y = split_data(class69x,class69y, 0.8)

    #1vs8
    class18x, class18y = class_versus_class(train[0], train[1], 1, 8)
    train18x, train18y, test18x, test18y = split_data(class18x,class18y, 0.8)

    #6vsAll
    class6x, class6y = class_versus_all(train[0], train[1], 6)
    train6x, train6y, test6x, test6y = split_data(class6x,class6y, 0.8)


    # perceptron6 = Lineaire(hinge_batch,hinge_g_batch,max_iter=1000,eps=0.1)
    # W6, C6 = perceptron6.fit_batch(train6x,train6y)
    # w_opt = W6[-1]
    # print(w_opt)
    # #Toutes les valeurs convergent vers une même valeur : dans ce cas ~ -3
    
    # te69 = []
    # te18 = []
    # te6 = []

    # for it in range(100):
    #     print("iter {}".format(it))
    #     perceptron69 = Lineaire(hinge_batch,hinge_g_batch,max_iter=it,eps=0.1)
    #     perceptron69.fit_batch(train69x,train69y)
    #     te69.append(perceptron69.score(test69x,test69y))

    #     perceptron18 = Lineaire(hinge_batch,hinge_g_batch,max_iter=it,eps=0.1)
    #     perceptron18.fit_batch(train18x,train18y)
    #     te18.append(perceptron18.score(test18x,test18y))

    #     perceptron6 = Lineaire(hinge_batch,hinge_g_batch,max_iter=it,eps=0.1)
    #     perceptron6.fit_batch(train6x,train6y)
    #     te6.append(perceptron69.score(test6x,test6y))
    
    # te69 = np.array(te69)
    # te18 = np.array(te18)
    # te6 = np.array(te6)

    # plt.figure()    
    # plt.plot(te69)
    # plt.savefig("6vs9.png")
    # plt.clf()

    # plt.figure()    
    # plt.plot(te18)
    # plt.savefig("1vs8.png")
    # plt.clf()

    # plt.figure()    
    # plt.plot(te6)
    # plt.savefig("6vsAll.png")
    # plt.clf()
    


    ''' 3 données 2D et projection '''
    testx1,testy1 =  gen_arti(nbex=10,data_type=0,epsilon=1)
    testx2,testy2 =  gen_arti(nbex=10,data_type=0,epsilon=1)
    print("Erreur : batch, test2 %f"% (perceptron_batch.score(testx1,testy1)))
    # #résultats très proches du premier test, normal, le perceptron n'a rien appris d'autre. C'est tiré de la même loi


    #projection polynomiale
    proj_x1 = projection_polynomiale(testx1, testx2)
    proj_y1 = projection_polynomiale(testy1, testy2) #pas dans le cours, mais x est forcément associé à y
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    Ws, Cs = perceptron.fit(proj_x1, proj_y1)

    plt.figure()
    plot_frontiere(proj_x1,perceptron.predict,200)
    plot_data(testx1,testy1)
    plot_data(testx2,testy2)
    plt.savefig("frontier_projected_datas") 
    
    ''' gauss '''
    noyau_g = noyau_gauss(testx1, testx2)
    print(noyau_g)
    #résultat: 0 ==> les 2 sont orthogonaux !
    # une exponentielle n'atteint 0 qu'à moins l'infini, et la fonction a une exponentielle négative donc strictement décroissante, alors il vaut mieux en avoir moins
    # les 2 vecteur de feature sont corrélé
