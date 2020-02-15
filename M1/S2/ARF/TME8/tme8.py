import numpy as np
import matplotlib.pyplot as plt


im=plt.imread("index.png")[ :, :, :3] #on garde que les 3 premieres composantes, la transparence est inutile
im_h, im_l ,_=im.shape
pixels=im.reshape((im_h*im_l , 3)) #transformation  en  matrice n∗3 , n nombre de pixels
imnew=pixels.reshape((im_h, im_l , 3)) #transformation  inverse
plt.imshow(im) #afficher l’image 
# plt.show()


def cout(data):
    
    pass

def Kmeans(data, n):
    mu = np.zeros( np.size(data,1) )
    for x in data:
        mu += x
    mu /= np.size(data,0)
    np.argmin(np.linalg.norm(mu-data[:])**2)  #sC(x)
    pass