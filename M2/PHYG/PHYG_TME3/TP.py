import numpy as np 

genome = {
    1: [1,0,2,3,4,5,6,7,8,-10,-9,11,-16,-15,-25,22,23,24,-21],
    2: [-20,0,-19,-18,-17,12,13,14]
}


def poisson(lam):
	val = 0
	while val == 0:
		val = int(np.random.poisson(lam, 1))
	return val

def choose_coordinates(g): 
    # it returns a pair (chrNum,chrPos), where chrNum identifies a (randomly chosen) chromosome of the input genome g and chrPos is a
    # (random) index over the list of genes g[chrNum].

    chrNum = np.random.choice( list(g.keys()) )
    sequence = g[chrNum]
    chrPos = np.random.randint( len(sequence) )

    return (chrNum,chrPos)

def inversion(g,mean_inv_len):
    # it returns the genome which can be obtained after
    # the application of an inversion event to the input genome g. The parameter
    # mean_inv_len defines the lambda parameter of the Poisson distribution used to choose
    # the number of genes involved in the inversion.

    #rand coord
    chro, pos = choose_coordinates(g)
    lon = poisson(mean_inv_len) 
    chromosome = g[chro]
    direction = 1
    try: 
        if( chromosome[pos+lon] ):
            section = chromosome[pos:pos+lon]
    except IndexError:
        if(pos-lon >= 0):
            section = chromosome[pos-lon:pos]
            direction = -1 
        else:
            print("retry...")
            return inversion(g, mean_inv_len)

    print(chromosome)
    print(section, lon, direction)
    section = section[::-1]
    section = - np.array(section)

    if(direction == 1):
        for i in range(lon):
            new_c = section[i]
            chromosome[pos+i] = new_c
    else:
        for i in range(lon):
            new_c = section[i]
            chromosome[pos-lon+i] = new_c

    print(chromosome)        
    g[chro] = chromosome    


def deletion(g,mean_del_len):
    # it returns the genome which can be obtained af-
    # ter the application of a deletion event to the input genome g. The parameter
    # mean_del_len defines the lambda parameter of the Poisson distribution used to choose
    # the number of genes involved in the deletion.

    #rand coord
    chro, pos = choose_coordinates(g)
    lon = poisson(mean_del_len) 
    chromosome = g[chro]
    direction = 1
    try: 
        if( chromosome[pos+lon] ):
            section = chromosome[pos:pos+lon]
    except IndexError:
        if(pos-lon >= 0):
            section = chromosome[pos-lon:pos]
            direction = -1 
        else:
            print("retry... out of bound section")
            return deletion(g,mean_del_len)

    if(0 in section): #cannot delete centromere
        print("retry... cannot delete centromere")
        return deletion(g,mean_del_len)

    print(chromosome)
    print(section, lon, direction)

    if(direction == 1):
        for i in range(lon):
            del chromosome[pos] 
    else:
        for i in range(lon):
            del chromosome[pos-lon]

    print(chromosome)        

def fission(g): 
# it returns the genome which can be obtained after the application of
# a fission event to the input genome g. 

    #rand coord
    chro, pos = choose_coordinates(g)
    chromosome = g[chro]
    del g[chro]

    ID = chro+0.1
    g[ID] = chromosome[:pos]
    ID += 0.1
    g[ID] = chromosome[pos:]

    

# inversion(genome, 5) 
# deletion(genome, 5)

fission(genome)
print(genome)