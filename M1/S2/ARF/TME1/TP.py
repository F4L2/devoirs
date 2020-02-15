import numpy as np
import pickle as pkl
from random import shuffle, randint, sample
import matplotlib.pyplot as plt
import itertools
from collections import deque

from  decisiontree import *

# data : tableau (films ,features), 
# id2titles : dictionnaire  id -> titre ,
# fields : id  feature  -> nom
[data , id2titles , fields ]= pickle.load(open("imdb_extrait.pkl","rb"))
# la  derniere  colonne  est le vote

###############################################################

#1.1
def entropie(vect):
    count = Counter(vect)
    prob = []
    for v in vect:
        prob.append( count[v] / len(vect) )
    H = -sum(prob[:] * np.log(prob[:]))
    return H

#1.2
def entropie_cond(liste_vect):
    probP = []
    len_all_sublist = 0
    for v in liste_vect:
        len_all_sublist += v.size
    for v in liste_vect:
        probP.append( v.size / len_all_sublist )
    H = 0.0
    it = 0
    for v in liste_vect:
        H += probP[it] * entropie(v)
        it += 1
    return H


#no problem with learning multiple times the same data. 
#shouldn't need labels.
#TODO : optimization : shuffle and slice the vector instead
def divide_db(data_base, label_base, percentage):
    data = data_base.copy()
    labels = label_base.copy()
    size = len(data)
    size_l = np.floor(size * percentage)
    size_t = np.floor(size - size_l)
    
    
    #data = shuffle(data)
    learn_x = [] 
    learn_y = []
    test_x = []
    test_y = []

    for i in range(np.asscalar((size_l).astype(int))):
        d = randint(0, size-1)
        learn_x.append( data[d] )
        learn_y.append( labels[d] )
    for i in range(np.asscalar((size_t).astype(int))):
        d = randint(0, size-1)
        test_x.append( data[d] )
        test_y.append( labels[d] ) 

    learn_x = np.array(learn_x)
    learn_y = np.array(learn_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return learn_x, learn_y, test_x, test_y


def get_score(learn_x, learn_y, test_x, test_y, depth):
    dt = DecisionTree ()
    
    sequence = range(1,depth)
    score_l = []
    score_t = []

    #learning
    for d in sequence:
        dt.max_depth = d
        dt.min_samples_split = 2 
        dt.fit(learn_x , learn_y)
        dt.predict(learn_x [:d ,:])

        score_l.append(1 - dt.score(learn_x , learn_y)) #dt.score = moyenne des bonne prédictions donc ==> 1 - dt.score = moyenne des mauvais prédictions 

    #testing
    for d in sequence:
        dt.max_depth = d
        dt.min_samples_split = 2
        dt.fit(test_x , test_y)
        dt.predict(test_x [:d ,:])

        score_t.append(1 - dt.score(test_x , test_y))

    print(score_l, score_t)
    return score_l, score_t


def cross_init( datax, datay, nb_chunks):
    cross_data = datax.copy()
    cross_label = datay.copy()

    both = list( zip(cross_data, cross_label) )
    shuffle(both)

    cross_data, cross_label = zip(*both)

    cross_data = np.array_split(cross_data, nb_chunks)
    cross_label = np.array_split(cross_label, nb_chunks)

    return cross_data, cross_label

def cross_reference(datax, datay, nb_chunks, depth):
    cross_data, cross_label = cross_init(datax, datay, nb_chunks)
    #order = sample(range(nb_chunks), nb_chunks)
    order = deque( [i for i in range(nb_chunks)] ) 

    learn_x = []
    learn_y = []
    test_x = []
    test_y = []

    score_learn = None
    score_test = None

    last_t = 1866 #valeur >1 quelconque
    last_l = 1866

    for all_comb in range(nb_chunks):
        subL_x = []
        subL_y = []
        subT_x = []
        subT_y = []
        for i in order:
            if( i == order[-1]):
                subT_x = list( cross_data[i] )
                subT_y = list( cross_label[i] )
                break
            subL_x = list( itertools.chain(subL_x, cross_data[i]) )
            subL_y = list( itertools.chain(subL_y, cross_label[i]) )

        order.rotate(1) #decalage vers la droite

        subL_x = np.array(subL_x)
        subL_y = np.array(subL_y)
        subT_x = np.array(subT_x)
        subT_y = np.array(subT_y)
        score_l, score_t = get_score(subL_x, subL_y, subT_x, subT_y, depth)

        learn_x.append( subL_x )
        learn_y.append( subL_y )
        test_x.append( subT_x )
        test_y.append( subT_y )

        #meilleures résultats
        if(score_t[-1] < last_t):
            score_test = score_t
            last_t = score_t[-1]
        if(score_l[-1] < last_l):
            score_learn = score_l
            last_l = score_l[-1]


    score_learn = np.array(score_learn)
    score_test = np.array( score_test )

    plt.plot(score_learn, label="Learn")
    plt.plot(score_test, label="Test")
    plt.savefig("cross_ref")
    plt.gcf().clear()
        




##################################################################################################
datax= data [: ,:32]
datay= np.array ([1 if x[33] >6.5  else  -1 for x in data])

#1.3

# features = data.copy().T
# entrop = []
# for xi in features:
#     entrop.append(entropie(xi))
# entrop_c = entropie_cond(features)
# entrop = np.array(entrop)
# # print(entrop)


# y_pos = np.arange(len(entrop))
# plt.bar(y_pos, entrop, align='center', alpha=1)
# plt.xticks(y_pos, y_pos, rotation=70)
# plt.ylabel('Feature')
# plt.title('Entropy')
# # plt.show()
# plt.savefig("entropy")
# plt.gcf().clear()


# entrop = entrop[:] - entrop_c
# print(np.argmax(entrop))
# print(entrop)
# y_pos = np.arange(len(entrop))
# plt.bar(y_pos, entrop, align='center', alpha=1)
# plt.xticks(y_pos, y_pos, rotation=70)
# plt.ylabel('Feature')
# plt.title('Entropy minus conditionnal')
# # plt.show()
# plt.savefig("entropy_minus_conditionnal")
# plt.gcf().clear()




# dt = DecisionTree ()
# dt.max_depth = 60 #on fixe la  taille  de l’arbre a 5
# dt.min_samples_split = 2 #nombre  minimum d’exemples  pour  spliter  un noeud
# dt.fit(datax ,datay)
# dt.predict(datax [:5 ,:])

# print(dt.score(datax ,datay)) 

# dt.to_pdf("test_tree.pdf",fields)      # dessine l’arbre  dans un  fichier  pdf   si pydot  est  installe.
# sinon  utiliser  http :// www.webgraphviz.com/dt.to_dot(fields)
#ou dans la  console
# print(dt.print_tree(fields ))


#1.4 : plus c'est profond, plus c'est long à calculer [je n'ai pas le visuel]
#       depth = 5 --> precision 0.73
#       depth = 6 --> precision 0.75
#       depth = 7 --> precision 0.77
#       depth = 8 --> precision 0.78

#1.5 : plus c'est profond, plus c'est précis. Oui, c'est normal.

#1.6 : A très grande profondeur, on est en train d'apprendre par coeur. 
#      On peut améliorer la fiabilité en ne testant pas sur les données d'apprentissage. 

###Il faut trouver la meilleure profondeur pour une bonne apprentissage. 

#1.7
#bleu = learn
#orange = test

depth = 20

# learn_x, learn_y, test_x, test_y = divide_db(datax, datay, 0.2)    #usage (data, percentage to learn)
# score_l, score_t = get_score(learn_x, learn_y, test_x, test_y, depth)

# plt.plot(score_l, label="Learn")
# plt.plot(score_t, label="Test")
# plt.savefig("learn02")
# plt.gcf().clear()

# learn_x, learn_y, test_x, test_y = divide_db(datax, datay, 0.5)    #usage (data, percentage to learn)
# score_l, score_t = get_score(learn_x, learn_y, test_x, test_y, depth)

# plt.plot(score_l, label="Learn")
# plt.plot(score_t, label="Test")
# plt.savefig("learn05")
# plt.gcf().clear()

# learn_x, learn_y, test_x, test_y = divide_db(datax, datay, 0.8)    #usage (data, percentage to learn)
# score_l, score_t = get_score(learn_x, learn_y, test_x, test_y, depth)

# plt.plot(score_l, label="Learn")
# plt.plot(score_t, label="Test")
# plt.savefig("learn08")
# plt.gcf().clear()


#1.8
# L'erreur progresse avec un 'pas' plus petit lorsqu'il y a plus de donnée à apprendre. 


#1.9
#bleu = test
# Validation croisée
nb_chunks = 6
# cross_reference(datax, datay, nb_chunks, depth)


