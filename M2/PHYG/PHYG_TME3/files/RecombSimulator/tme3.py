# -*- coding: Utf-8 -*-

# developped by Aubin Fleiss
# contact : aubin.fleiss@gmail.com

import random as rd
import numpy as np
import copy
import sys

""" This function return a non-zero value according to a Poisson distribution
with the lambda parameter provided in input """
def poisson(lam):
	val = 0
	while val == 0:
		val = int(np.random.poisson(lam, 1))
	return val

""" This function takes in input a genome and returns a pair where
the first value is a chromosome identifier and the second value is a random
index within the list of genes of the choosen chromosome """
def choose_coordinates(genome):
	# choose chromosome : adjust probability based on chromosome length
	chroms=genome.keys()
	nbGenes=sum([len(genome[chrom]) for chrom in genome])
	p=[float(len(genome[chrom]))/float(nbGenes) for chrom in sorted(genome.keys())]
	chromChoice = int(np.random.choice(sorted(genome.keys()), 1, p=p))
	# choose gene in previously selected chromosome
	geneChoice = rd.choice(range(len(genome[chromChoice])))
	return((chromChoice,geneChoice))

""" This function takes in input a genome and prints it to the standard output """
def print_genome(genome):
	for chrom in genome.keys():
		print(chrom),
		print(genome[chrom])


def inversion(genome,mean_inv_len):
	""" EXERCISE 2 - Inversion function """
	#rand coord
	chro, pos = choose_coordinates(genome)
	lon = poisson(mean_inv_len)
	chromosome = genome[chro]
	direction = 1
	try:
		if( chromosome[pos+lon] ):
			section = chromosome[pos:pos+lon]
	except IndexError:
		if(pos-lon >= 0):
			section = chromosome[pos-lon:pos]
			direction = -1 
		else:
			print("retry... section out of bound")
			return inversion(genome, mean_inv_len)
	try: 
		section = section[::-1]
		section = - np.array(section)
	except UnboundLocalError:
		print("retry...something has gone wrong")
		return inversion(genome, mean_inv_len)
	
	if(direction == 1):
		for i in range(lon):
			new_c = section[i]
			chromosome[pos+i] = new_c
	else:
		for i in range(lon):
			new_c = section[i]
			chromosome[pos-lon+i] = new_c
			
	genome[chro] = chromosome    
	return(genome)


def deletion(genome,mean_del_len):
	""" EXERCISE 2 - Deletion function """
	#rand coord
	chro, pos = choose_coordinates(genome)
	lon = poisson(mean_del_len) 
	chromosome = genome[chro]
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
			return deletion(genome,mean_del_len)
	
	if(0 in section): #cannot delete centromere
		print("retry... cannot delete centromere")
		return deletion(genome,mean_del_len)
	
	if(direction == 1):
		for i in range(lon):
			del chromosome[pos] 
	else:
		for i in range(lon):
			del chromosome[pos-lon]

	genome[chro] = chromosome
	return(genome)


def fission(genome):
	""" EXERCISE 2 - Fission function """
	#rand coord
	chro, pos = choose_coordinates(genome)
	chromosome = genome[chro]
	del genome[chro]
	
	ID = chro+0.1
	new_chrom = chromosome[:pos]
	if(0 not in new_chrom):
		new_chrom.insert( int(np.ceil(len(new_chrom)/2)) , 0 )
	genome[ID] = new_chrom
	ID += 0.1
	new_chrom = chromosome[pos:]
	if(0 not in new_chrom):
		new_chrom.insert( int(np.ceil(len(new_chrom)/2)) , 0 )
	genome[ID] = new_chrom
	return(genome)


def main( argv=None ):

	# definition of a test genome
	genome = {}
	genome[1]=range(1,22)
	genome[1].insert(5,0)
	genome[2]=range(22,54)
	genome[2].insert(10,0)
	genome[3]=range(54,76)
	genome[3].insert(7,0)
	genome[4]=range(76,101)
	genome[4].insert(11,0)

	print("### ancestor genome ###")
	print_genome(genome)

	print("### genome after an inversion ###")
	genome = inversion(genome,5)
	print_genome(genome)

	print("### genome after a deletion ###")
	genome=deletion(genome,1)
	print_genome(genome)

	print("### genome fission ###")
	genome=fission(genome)
	print_genome(genome)

	return 0


if __name__=="__main__":
	sys.exit(main())
