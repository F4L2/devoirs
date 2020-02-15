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


from markov_tools import *




data = pkl.load(open("genome_genes.pkl","rb"), encoding='latin1')

Xgenes  = data.get("genes") #Les genes, une array de arrays

Genome = data.get("genome") #le premier million de bp de Coli

Annotation = data.get("annotation") ##l'annotation sur le genome
##0 = non codant, 1 = gene sur le brin positif

### Quelques constantes
DNA = ["A", "C", "G", "T"]
stop_codons = ["TAA", "TAG", "TGA"]



##############################################

a = 1/200

mb = 0
for i in range( len(Xgenes)):
    mb+= len( Xgenes[i] )
mb = mb / len(Xgenes)

b = 1/mb

'''
Pi = np.array([1, 0, 0, 0])
A =  np.array([[1-a, a  , 0, 0],
              [0  , 0  , 1, 0],
              [0  , 0  , 0, 1],
              [b  , 1-b, 0, 0 ]])
B = . . .

# pour utiliser le modèle plus loin:
s, logp = viterbi(Genome,Pi,A,B)
'''

inter_A = (Genome == 0).sum()
inter_C = (Genome == 1).sum()
inter_G = (Genome == 2).sum()
inter_T = (Genome == 3).sum()

Binter = np.zeros( (1,4) )
Binter[0][0] = inter_A  / len(Genome)
Binter[0][1] = inter_C  / len(Genome)
Binter[0][2] = inter_G  / len(Genome)
Binter[0][3] = inter_T  / len(Genome)

print(Binter)

Bgene = np.zeros( (3,4) )

codon = []
sumg = 0
for g in Xgenes:
        #codon = [ Xgenes[i][0], Xgenes[i][1], Xgenes[i][2] ]
        for j in range( 3, len(g) - 3, 3 ) :
                Bgene[0][g[j]] += 1
                Bgene[1][g[j+1]] += 1
                Bgene[2][g[j+2]] += 1
        sumg += len(g) - 6


for i in range ( len(Bgene) ):
        for j in range( len(Bgene[i]) ):
                Bgene[i][j] /= sumg / 3

print(Bgene)


B_m1 = np.vstack((Binter, Bgene))


B_m2 = [ [1-a, a*0*83, a*0.13, a*0.03, 0, 0, 0, 0, 0, 0, 0, 0], #1-a revenir inter, a*83% ATG, a*13% GTG, a*3% TTG
         [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #ATG
         [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #GTG
         [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #TTG
         [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #codon 0
         [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #codon 1
         [ 0, 0, 0, 0, 1-b, 0, 0, b, 0, 0, 0, 0], #codon 2
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0], #T 
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5], #A
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #G
         [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #G
         [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  #A
        ]


'''
model1._set_emissionprob(B_m1)

vsbce, pred = model1.decode(Genome)
#vsbce contient la log vsbce
#pred contient la sequence des etats predits (valeurs entieres entre 0 et 3)

#on peut regarder la proportion de positions bien predites
#en passant les etats codant a 1
sp = pred
sp[np.where(sp>=1)] = 1
percpred1 = float(np.sum(sp == Annotation) )/ len(Annotation)

percpred1
Out[10]:  0.636212
'''



'''
f= open('data.pkl','wb')
pkl.dump(data,f) # penser à sauver les données pour éviter de refaire les opérations
f.close()
'''
