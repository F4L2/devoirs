'''
N° étudiant: 3870665
'''
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')   
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb
from mpl_toolkits.mplot3d import Axes3D

import time
import pickle as pkl


def read_file ( filename ):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )    
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty ( 10, dtype=object )  
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )

    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( list ( map ( lambda x: float(x), champs ) ) )
    infile.close ()

    # transformation des list en array
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )

    return output

def display_image ( X ):
    """
    Etant donné un tableau X de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )

    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)

    # affichage de l'image
    plt.imshow( img )
    plt.show ()


def learnML_class_parameters ( imgs ):
    #img[n_image][256_pixels] 
    data = np.zeros((2,256)) #0 : mu, 1: sigma²

    for i in range ( 256 ):
        data[0][i] = imgs[:,i].mean() #imgs[:,i].sum() / len(imgs)
        data[1][i] = pow( imgs[:,i].std(), 2) #pow( (imgs[:,i] - data[0,i]), 2 ).sum() / len(imgs)

    #transform output list into array
    output  = np.array( data ) # /!\

    return output

def learnML_all_parameters ( data ):
    output = np.zeros( (len(data), 2, 256) )

    for i in range ( len(data )):
        output[i] = learnML_class_parameters ( training_data[i] )

    return output

def log_likelihood ( img, param ):
    log = 0
    for i in range (256):
        if(param[1,i] != 0):
            log += ( -0.5 * np.log( 2 * np.pi * param[1,i]) ) - ( 0.5 * ( pow((img[i] - param[0,i]), 2) / param[1,i]) )
    return log

def log_likelihoods ( img, param ):
    output = np.zeros( (10) )

    for i in range(10) :
        output[i] = log_likelihood( img, param[i])

    return output 

def classify_image ( img, param ):
    log = log_likelihoods(img, param)
    return log.argmax()

def classify_all_images (imgs, param):
    T = np.zeros( (10,10) )

    for i in range ( 10 ):
        l = []
        for k in range( len(imgs[i])):
            l.append( classify_image(imgs[i][k], param) )
        for j in range ( 10 ):
            T[i,j] = list(l).count(j)
            T[i,j] /= len(l)

    return T


def dessine ( classified_matrix ):
    fig = plt.figure()
    plt.imshow(classified_matrix)
    #ax = fig.add_subplot(111, projection='3d')
    #x = y = np.linspace ( 0, 9, 10 )
    #X, Y = np.meshgrid(x, y)
    #ax.plot_surface(X, Y, classified_matrix, rstride = 1, cstride=1 )
    plt.show()

#######################################################################

training_data = read_file("train.txt")
#display_image ( training_data[2][0] )

par = learnML_class_parameters(training_data[0])

parameters = learnML_all_parameters(training_data)
test_data = read_file ( "test.txt" )

#log = log_likelihood ( test_data[2][3], parameters[1] )
#log = log_likelihoods ( test_data[1][5], parameters )

num = classify_image( test_data[1][5], parameters )
#print(num)
num = classify_image( test_data[4][1], parameters )
#print(num)

T = classify_all_images ( test_data, parameters )
dessine(T)

'''
f= open('data.pkl','wb')
pkl.dump(data,f) # penser à sauver les données pour éviter de refaire les opérations
f.close()
'''
