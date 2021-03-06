import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from copy import deepcopy

from tools.parse_pdb import readPDB

pdb_file1 = "3pdz.pdb"
pdb_file2 = "1fcf_aliSeq.pdb"

exp, resol, nummdl, lesChaines, lesAtomes1 = readPDB(pdb_file1, 'A') 
exp, resol, nummdl, lesChaines, lesAtomes2 = readPDB(pdb_file2, 'A') 





### A. RSMD

sel1 = [21 , 22 , 23 , 24 ,      26,       28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,       53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68]
sel2 = [158, 159, 160, 161, 162,      164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,                          183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208]

num_res1 = {}
num_res2 = {}
for a in lesAtomes1:
    num_res1[a.numRes] = a

for a in lesAtomes2:
    num_res2[a.numRes] = a


def RMSD(sel1,sel2,num_res1,num_res2):
    somme = 0
    n = 0
    for ind1,ind2 in zip(sel1,sel2):
        n += 1
        a1 = num_res1[ind1]
        a2 = num_res2[ind2]
        somme += (a1.x - a2.x)**2 + (a1.y - a2.y)**2 + (a1.z - a2.z)**2
    return np.sqrt( somme / n ) 

#print( RMSD(sel1,sel2,num_res1,num_res2) )

'''
RMSD: 10.46
Non les 2 résidus alignés ne sont pas superposés
On peut déduire que l'alignement n'est pas parfait
'''

### B. contact map

def distance(a1, a2):
    return np.sqrt( (a1.x-a2.x)**2 + (a1.y-a2.y)**2 + (a1.z-a2.z)**2 )

def contact_map(sel1,sel2,num_res1,num_res2):
    c_mat = np.zeros((len(sel1), len(sel2)))
    for i in range(len(sel1)):
        a1 = num_res1[sel1[i]]
        for j in range(len(sel2)):
            a2 = num_res2[sel2[j]]
            c_mat[i,j] = distance(a1,a2)

    plt.figure()
    plt.pcolor(c_mat)
    #plt.show()
    plt.savefig("contact_map.png")

# contact_map(sel1,sel2,num_res1,num_res2)

'''
Pour la dissimilarité on peut juste inverser le signe des valeurs de contact. 
'''


### C. circular variance
def circular_variance(lesAtomes, rayon = 20):
    residus_score = {}
    for a1 in lesAtomes:
        somme_vec = 0 
        for a2 in lesAtomes:
            if(a1==a2 or distance(a1,a2) > 20):
                continue

            r = [a2.x-a1.x, a2.y-a1.y, a2.z-a1.z] #vecteur
            somme_vec += r / np.linalg.norm(r)
        somme = np.linalg.norm(somme_vec)
        score = 1 - somme / len(lesAtomes)

        residus_score[ a1.numRes ] = score

    return residus_score

    #TODO: calculer CV à partir des atomes et rayon en parametre => ne pas donner CV en parametre
def enfouis(cv , x= 0.1):
    size = int(len(cv) * x)
    sorted_cv = sorted(cv.items(), key=lambda x: x[1]) #becomes list of tuple
    top = [sorted_cv[i] for i in range(size)]
    return top 

def protuberants(cv , x= 0.1):
    size = int(len(cv) * x)
    sorted_cv = sorted(cv.items(), key=lambda x: x[1], reverse=True) #becomes list of tuple
    top = [sorted_cv[i] for i in range(size)]
    return top 

# cv1 = circular_variance(lesAtomes1)

# #10% les plus enfouis/protuberants
# print( enfouis(cv1) )
# print( protuberants(cv1) )


# TODO: juste réécrire les CV ? Score à la colonne 12 => pas de sens
#       comprendre la question
def ATOMcol12(pdb_f, CV): #lis et écris les CV dans la 12eme col d'un fichier pdb
    out_f = pdb_f.split(".")[0]+'_modified.pdb'
    out = open(out_f,'w')

    dictionnaire = {}
    with open(pdb_f, 'r') as f:
        for line in f:
            if(line[:4] != "ATOM"):
                out.write(line)
                continue
            columns = line.strip().split(' ')
            cols = list(filter(None, columns))
            ID = int(cols[5])

            try: 
                columns[-1] = str(CV[ID])  
            except KeyError:
                continue            #not carbon alpha

            n_line = ""
            for s in columns:
                if(s==''):
                    n_line+=' '
                else:
                    n_line += s
            n_line +=" \n"
            out.write(n_line)
    out.close()
    return dictionnaire


# C.4
#separate chains A,B
pdb_f = "2bbm.pdb"
exp, resol, nummdl, lesChaines, AtomesA = readPDB(pdb_f, 'A')
cvA = circular_variance(AtomesA)
ATOMcol12(pdb_f, cvA)
exp, resol, nummdl, lesChaines, AtomesB = readPDB(pdb_f, 'B')
cvB = circular_variance(AtomesB)

#combined AB
max_nResA = max(a.numRes for a in AtomesA)
AtomesAB = AtomesA.copy()
for aB in AtomesB:
    a = deepcopy( aB )
    a.numRes += max_nResA
    AtomesAB.append(a)
cvAB = circular_variance(AtomesAB)

# print(enfouis(cvA))
# print(protuberants(cvA))
# print()
# print(enfouis(cvB))
# print(protuberants(cvB))
# print()
# print(enfouis(cvAB))
# print(protuberants(cvAB))

'''
La chaine A a possède à la fois des résidus plus enfouis et plus protubérantes que la chaines B
La chaine B étant courte, il n'y a que peu de différence avec A lorsque l'on compare avec les 2 chaines combinées. 
'''

### D. champ de force
from tools.ForceField import chargePDB, epsilon_vdw_PDB

dcharge = chargePDB()
dvdw, depsilon = epsilon_vdw_PDB()
f = 332.0522


# énergie covalente/bonded déjà dans le dico

def non_cov(R,L):
    # A?? 
    # B?? 
    # => R? no L? 
    pass

def e_tot(P): #protéine dans le vide
    for a1 in P:
        for a2 in P:
            if( max(a1.numRes,a2.numRes) - min(a1.numRes,a2.numRes) >= 4 ): #covalent
                dcharge[a1.resType][a2.resType]
                pass
            else: #non-covalent
                pass
    return 

def e_tot_duet(R,L):
    return e_tot(R) + e_tot(L) + non_cov(R,L)

def e_interaction(R,L):
    return e_tot_duet(R,L) - e_tot(R) - e_tot(L)

#chaine A
# AtomesA

#chaine B
# AtomesB 





# E. elastic network model 

F = 1.0
def atom_linkage(atoms, cutoff):
    links = []
    for a1 in atoms:
        for a2 in atoms:
            if(distance(a1,a2) < cutoff ):
                links.append( (a1,a2) )
    return links


def network(pdb_file, new_pdb_file, cutoff = 5):
    exp, resol, nummdl, lesChaines, atoms = readPDB(pdb_file, 'A')
    links = atom_linkage(atoms, cutoff)

    dic = {}
    for a1, a2 in links:
        if(a1.numRes not in dic):
            dic[a1.numRes] = [a2.numRes]
        else:
            dic[a1.numRes].append(a2.numRes)

    out = open(new_pdb_file, "w")
    with open(pdb_file, "r") as f:
        lines = f.readlines()
        length = len(lines)
        body = lines[:-2] #master and end
        footer = lines[length-2:]

        for line in lines:
            out.write(line)
        
        for k,v in dic.items():
            string = str(k)+" "
            for res in v:
                string+=str(res)+' '
            line = 'CONNECT\t' + string +'\n'
            out.write(line)

        for line in footer:
            out.write(line)


network(pdb_file = "3pdz.pdb", new_pdb_file= "ElasticNet.pdb", cutoff= 5)