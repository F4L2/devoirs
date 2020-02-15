#spanish_inquisition, maxsize, blabla

import numpy as np 
import itertools as it

spanish_inquisition = 1000




def formatage (matrice):
        mat = matrice.copy().tolist()
        for i in range(len(mat)):
                mat[i].pop(i)

        return np.array(mat)


def init_cl(size):
    cl = []
    for i in range(size):
        cl.append( chr(ord("A") + i) )
    return cl

def update_cluster(cl, cluster_gauche, cluster_droite):

    cluster_union = []
    cluster_union.append(cluster_gauche)
    cluster_union.append(cluster_droite)
    '''
    for c in cluster_gauche:
            cluster_union.append(c)
    for c in cluster_droite:
            cluster_union.append(c)      
    '''
    cl.append(cluster_union)
    cl.remove(cluster_gauche)
    cl.remove(cluster_droite)

    return cl


def update_matrix(mat, li_m, co_m, new_line, new_col):
    
    #remove previous clusters, higher index first
    first = max( co_m, li_m )
    second = min( co_m, li_m )
    mat = np.delete(mat, first, 1)
    mat = np.delete(mat, first, 0)
    mat = np.delete(mat, second, 1)
    mat = np.delete(mat, second, 0)

    
    #add new joigned cluster in distance matrix
    mat = np.vstack( (mat, new_line) )
    mat = np.hstack( (mat, new_col) )

    return mat


def neighbor(mat):
    size = len(mat)
    U = np.zeros( size )
    for i in range(size):
        U[i] = (mat[i].sum() - mat[i][i]) / (size - 2)  ########## ERROR! size = 2 
    return U

def distanceToNeighbor(mat, U):
    size = len(mat)
    Q = np.zeros( (size,size) )
    for i in range(size):
            for j in range(size):            
                Q[i][j] = mat[i][j] - U[i] - U[j]
    return Q


def new_clust_upgma(mat, li_m, co_m, cluster_gauche, cluster_droite):
    size = len(mat)
    new_li = []
    new_co = [spanish_inquisition] * (size-1)         #que des zeros, ou même val que new_li si tableau symetrique
    for i in range(size):
            if(i == li_m or i == co_m):
                    continue
            if(mat[i][li_m] == spanish_inquisition):
                    dgl = mat[li_m][i]
            else:
                    dgl = mat[i][li_m]
            if(mat[i][co_m] == spanish_inquisition):
                    ddl = mat[co_m][i]
            else:
                    ddl = mat[i][co_m]

            new_li.append( (len(cluster_gauche) * dgl + len(cluster_droite) * ddl) / (len(cluster_gauche) + len(cluster_droite)) )

    new_li = np.array(new_li)
    new_co = np.array(new_co)
    return new_li, new_co



def longWay(mat, cluster_gauche, cluster_droite):
    distGD = 0
    for p in cluster_gauche:
            for q in cluster_droite:
                    if(mat[p][q] == spanish_inquisition):
                            distGD += mat[q][p]
                    else:
                            distGD += mat[p][q]
    distGD = distGD / ( len(cluster_gauche) * len(cluster_droite) )
    return distGD


def combine (cl_g, cl_d):
        return "("+cl_g+", "+cl_d+")"

def add_weight(cl, weight):
        return cl+": "+str(weight)

def stringify( cl ):
        if( len(cl) == 1):
                return cl
        else:
                comp1 = stringify(cl[0])
                comp2 = stringify(cl[1])
                return combine( comp1, comp2 )


def branchify( cl, tree ):
        cumul = 0
        if( len(cl) == 1):
                output = cl
                if (cl in tree):
                        output = add_weight(cl, tree[cl])
                        cumul = tree[cl]
        else:
                key = stringify( cl )
                #print(key)
                if( key in tree):
                        return add_weight(key, tree[key])

                comp1, cumul1 = branchify( cl[0], tree)
                if( comp1 in tree ):
                        weight = tree[comp1] #- cumul1
                        #print(cumul1, " ----- ", weight)
                        cumul1 += weight
                        comp1 = add_weight( comp1, weight) 
                        
                        
                comp2, cumul2 = branchify( cl[1] , tree)  
                if( comp2 in tree ):
                        weight = tree[comp2] #- cumul2
                        cumul2 += weight
                        comp2 = add_weight( comp2, weight) 
                        
                output = combine(comp1, comp2)
                cumul = max(cumul1, cumul2)

        return output, cumul



def newick_forme(cl_g, cl_d, tree):
        comp1, cumul1 = branchify(cl_g, tree)
        comp2, cumul2 = branchify(cl_d, tree)

        if( comp1 in tree):
                comp1 = add_weight(comp1, tree[comp1] - cumul1)
        if( comp2 in tree):
                comp2 = add_weight(comp2, tree[comp2] - cumul2)
        return comp1, comp2, cumul1, cumul2


def build_tree(height, cl_g, cl_d, tree):
        br1, br2, cumul1, cumul2 = newick_forme(cl_g, cl_d, tree)
        
        tree[br1] = height - cumul1
        tree[br2] = height - cumul2

        return br1, br2


def build_neighbor_tree(height1, height2, cl_g, cl_d, tree):
        br1, br2, devnull, devnull = newick_forme(cl_g, cl_d, tree)
        
        tree[br1] = height1
        tree[br2] = height2

        return br1, br2

