import numpy as np


def init_dic( file ):
    dic = {}
    f = open( file , "r")
    for line in f:
        line = line[:-1]
        n1, n2 = line.split(',')
        #n1 = int(n1)
        #n2 = int(n2)

        if( n1 not in dic):
            dic[n1] = [ n2 ]
        else:
            dic[n1].append( n2 )

        if( n2 not in dic):
            dic[n2] = [ n1 ]
        else:
            dic[n2].append( n1 )
    f.close()
    return dic

def deg(dic):
    degree = np.zeros(len(dic))
    i = 0
    for key in dic:
        degree[i] = len(dic[key])
        i += 1

    print(degree)
    print( len(degree) )
    return degree

def loglog( degree ):
    #plot avec log du degré
    return 

def clustcoef ( degree ):
    
