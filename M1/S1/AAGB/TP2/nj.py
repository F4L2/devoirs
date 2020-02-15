import numpy as np 
import sys

from junk import *

#spanish_inquisition = sys.maxsize <=> 0 

def nj (mat):
    #init 
    size = len(mat)
    cl = init_cl(size)
    tree = {}

    #iteration
    while( size > 2 ):
        
        U = neighbor(mat) ### BUG: size == 2 
        Q = distanceToNeighbor(mat, U)
        
        #minimal Q
        arg_min = np.argmin(Q) 
        li_m = int(arg_min / size)
        co_m = arg_min % size

        distGD = mat[li_m][co_m]

        #find new dist
        new_line = []
        for i in range( len(mat) ):
            if( i == li_m or i == co_m ):
                continue  
            new_line.append( (mat[li_m][i] + mat[i][co_m] - distGD) / 2 ) 
        
        new_col = new_line.copy()
        new_col.append(spanish_inquisition)  
        new_col = np.array(new_col)
        new_col = new_col.reshape( (len(new_col),1) )
        new_line = np.array(new_line)

        mat = update_matrix(mat, li_m, co_m, new_line, new_col)
        size = len(mat)

        #update clusters
        cl_g = cl[li_m]
        cl_d = cl[co_m]
        cl = update_cluster(cl, cl_g, cl_d)

        #compute dist of tree branch
        dg = (distGD + U[li_m] - U[co_m]) / 2
        dg = round(dg, 4)
        dd = (distGD + U[co_m] - U[li_m]) / 2
        dd = round(dd, 4)

        #build tree
        br1, br2 = build_neighbor_tree(dg, dd, cl_g, cl_d, tree)

    cl_g = cl[0]
    cl_d = cl[1]
    cl = update_cluster(cl, cl_g, cl_d)

    distGD = mat[1][0]
    dg = round(distGD, 4)
    dd = round(distGD, 4)

    #build tree
    br1, br2 = build_neighbor_tree(dg, dd, cl_g, cl_d, tree)

    br1 = add_weight(br1, dg)
    br2 = add_weight(br2, dd)

    newick = combine(br1,br2)

    f = open("results/neighbor_joining.txt", "w")
    f.write(newick)
    f.close()

    return newick