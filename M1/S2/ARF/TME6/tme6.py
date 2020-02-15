import sklearn
from arftools import *

from sklearn.neighbors import KNeighborsClassifier
'''https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'''

from sklearn.svm import SVC
'''https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html'''
'''https://scikit-learn.org/stable/modules/svm.html'''


def  plot_frontiere_proba(data ,f ,step =20):
    grid, x, y= make_grid(data=data ,step=step)
    plt.contourf (x , y, f(grid).reshape(x.shape), 255)
    

######################################################################################################################################################################

# #(7291,256)(7291,)
train = load_usps("USPS/USPS_train.txt")
# #(2007,256)(2007,)
test = load_usps("USPS/USPS_test.txt")

trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=1)

# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(trainx, trainy) 
# print(neigh.predict([testx[0]]))
# print(neigh.predict_proba([testx[0]]))
# print(neigh.score(testx, testy, sample_weight=None))

'''USPS''' 
#BUG
# svm = SVC(gamma= 'auto', probability = True, kernel="linear")
# svm.fit(train[0],train[1])
# plot_frontiere_proba (train[0] ,lambda x : svm.predict_proba(x)[:,0], step =50)
# plt.savefig("USPS_lin")

# svm = SVC(gamma= 'auto', probability = True, kernel="poly")
# svm.fit(train[0],train[1])
# plot_frontiere_proba (train[0] ,lambda x : svm.predict_proba(x)[:,0], step =50)
# plt.savefig("USPS_pol")

# svm = SVC(gamma= 'auto', probability = True, kernel="rbf")
# svm.fit(train[0],train[1])
# plot_frontiere_proba (train[0] ,lambda x : svm.predict_proba(x)[:,0], step =50)
# plt.savefig("USPS_rbf")


'''arti'''

svm = SVC(gamma= 'auto', probability = True, kernel="linear")
svm.fit(trainx,trainy)
plot_frontiere_proba (testx ,lambda x : svm.predict_proba(x)[:,0], step =50)
plt.savefig("arti_lin")
print("en linéaire")
print(svm.n_support_)
# print(svm.support_vectors_)
# print(svm.coef_)        #alpha, seulement en linéaire
# print(svm.dual_coef_)   #beta
print(svm.fit_status_)

svm = SVC(gamma= 'auto', probability = True, kernel="poly")
svm.fit(trainx,trainy)
plot_frontiere_proba (testx ,lambda x : svm.predict_proba(x)[:,0], step =50)
plt.savefig("arti_pol")
print("\nen polynomiale")
print(svm.n_support_)
# print(svm.support_vectors_)
# print(svm.dual_coef_)
print(svm.fit_status_)

svm = SVC(gamma= 'auto', probability = True, kernel="sigmoid")
svm.fit(trainx,trainy)
plot_frontiere_proba (testx ,lambda x : svm.predict_proba(x)[:,0], step =50)
plt.savefig("arti_sig")
print("en sigmoide")
print(svm.n_support_)
# print(svm.support_vectors_)
# print(svm.dual_coef_)  
print(svm.fit_status_)

svm = SVC(gamma= 'auto', probability = True, kernel="rbf")
svm.fit(trainx,trainy)
plot_frontiere_proba (testx ,lambda x : svm.predict_proba(x)[:,0], step =50)
plt.savefig("arti_rbf")
print("\nen rbf")
print(svm.n_support_)
# print(svm.support_vectors_)
# print(svm.dual_coef_)
print(svm.fit_status_)


# si l'on prend le noyau linéaire comme référence, on a plus de vecteur support en polynomiale et rbf, mais moins en sigmoïdale. Quasiement la même quantité pour les 2 classes. 
# c'est normal et proportionnelle au nombre de dimension (?) 
# dans le cas linéaire on a accès aux coef alphas, qui nous indique les points près des vecteurs support. 


''' grid search '''
from sklearn.model_selection import GridSearchCV

# parameters = { 'gamma': ('auto','auto','auto','auto'), 'probability': (True,True,True,True) , 'kernel':('linear', 'poly', 'sigmoid', 'rbf')}
# svr = SVC()
# clf = GridSearchCV(svr, parameters , cv = 5)
# clf.fit(trainx, trainy)

# plot_frontiere_proba (trainx ,lambda x : clf.predict_proba(x)[:,0], step =50)
# plt.savefig("optim_train")

# plot_frontiere_proba (testx ,lambda x : clf.predict_proba(x)[:,0], step =50)
# plt.savefig("optim_test")



# la frontière en test est un peu décalé par rapport à celle en train. Il y a une petite région inconnue en blanc.
 

''' string kernel '''

import re
mot = 'aazefsbfgsdlfqsmdlqfekqerlzakemrqrljgkmsdflgkqsdaddfsdfsnjedsmlfbklfdklsadfklfkfkfovocdfsdfsefzezrerqs'
second_mot = 'erazeazfdgfcvxlqfekqerlzakemrqrljgkmsdflgkqsdaddfsdfsazearezrteazazeazasdqaazefsbfgsdld'
sous_mots = ['fsbfgsdlfq', 'azefsb', 'ljgkmsdflgkqsd', 'qerlzakemr', 'fsnjedsmlfbklfdkl', 'vocdfsdfsefzez']
lamb = 0.66

def find_occ(s,u):
    seqs = []
    motif=False
    start = 0
    it = 0
    for i in range(len(s)):
        if(motif):
            if(it > len(u)):
                if(s[i] == u[it]):
                    it+=1
                else:
                    motif = False
                    it = 0
            else:
                motif = False
                it = 0
                seqs.append(start)
            
        else:
            if(s[i] == u[0]):
                start = i
                motif = True 
    return seqs

def phi(s ,u, lam):
    u_in_s = find_occ(s, u)    
    return (lam**len(u))*len(u_in_s)

def string_kernel(s, t, sub, lam):
    res = 0
    for u in sub:
        res += phi(s,u,lam) * phi(t,u,lam)
    return res


res = string_kernel(mot, second_mot, sous_mots, lamb)
print(res)

# res = 0.33, les 2 mots sont assez proches