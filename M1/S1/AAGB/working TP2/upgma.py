import numpy as np
import sys

from junk import *

# spanish_inquisition <=> sys.maxsize pour rester lisible

def upgma(mat):
        #init
        size = len(mat)
        cl = init_cl(size)
        tree = {}
        
        while( len(cl) > 1 ):

                #find min
                arg_min = np.argmin(mat) 
                li_m = int(arg_min / size)
                co_m = arg_min % size

                #calcul distance entre cluster gauche et cluster droite (di,j)
                # distGD = longWay(mat,cl[li_m],cl[co_m])
                distGD = mat[li_m][co_m]
                
                #find new dist (dk,l)
                new_line, new_col = new_clust_upgma(mat, li_m, co_m, cl[li_m], cl[co_m])
                new_line.shape = ( (1, len(new_line)) )
                new_col.shape = ( (len(new_col), 1) )

                mat = update_matrix(mat, li_m, co_m, new_line, new_col)
                size = len(mat)
                
                #update clusters
                cl_g = cl[li_m]
                cl_d = cl[co_m]
                cl = update_cluster(cl, cl_g, cl_d)

                #build tree
                '''
                build newick tree here 
                '''
                height = distGD/2
                br1, br2 = build_tree(height, cl_g, cl_d, tree)
        
        br1 = add_weight(br1, tree[br1])
        br2 = add_weight(br2, tree[br2])

        newick = combine(br1,br2)
        
        return newick