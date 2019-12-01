# import pickle as pkl
import numpy as np
import pickle as pkl
from Bio import SeqIO, AlignIO
import subprocess
import os 

import ray
import psutil

num_cpus = psutil.cpu_count(logical=True)
ray.init(num_cpus=num_cpus)

# input : [blast output, "/data/output_blast.txt"]
# output : dic 


# format dic[ 'infos' ] : { header, liste séquences homologues }
# format dic[ sequence_ID ] : { field_data }

#@ray.remote(num_return_vals=2)
def parse(blast_output, iterations = 3):
    print(blast_output)
    convergence = False    
    dic = {}
    with open(blast_output, 'r') as f:
        head = [next(f) for x in range(6)]
        description = head[1].split(":")[1].strip()
        fields = head[4].split(":")[1].strip()
        caracteres = []
        for c in fields.split(","):
            caracteres.append(c.strip())
        caracteres = caracteres[2:]

        all_subjects = []
        subjects = []
        for l in f:
            if("Search has CONVERGED!" in l):
                convergence = True
                break

            if("# PSIBLAST 2.6.0+" in l):
                all_subjects.append(subjects)
                subjects = []

                for i in range(5): #sauter le header
                    l = next(f)
                l = next(f)        

            data = l.strip().split("\t")

            if(len(data)==1): #line is empty, search has converged message, end of file
                continue

            ID = data[1]
            dic[ID] = { }
            subjects.append(ID)
            data = data[2:]
            for c,d in zip(caracteres,data):
                dic[ID][c] = d    
                
    all_subjects.append(subjects)
    dic['infos'] = {"description": description, "subjects": all_subjects[-1]} #save last iteration
    return dic, convergence



def initialize(seqDir, resDir, inputFile): #, dbFile, dbDir):
    if not os.path.exists(resDir):
        os.mkdir(resDir)
    if not os.path.exists(seqDir):
        os.mkdir(seqDir)

    ##formatting db into blast format
    #if not os.path.exists(dbDir):
    #    subprocess.run(list(str("makeblastdb -in "+dbFile+" -parse_seqids -title database_local -dbtype prot -out "+dbDir).split(' ')))
    #    print("DB formatted ")
    #else:    
    #    print("DB already exists")

    for record in SeqIO.parse(inputFile, "fasta"):
        #print(record.id, record.description)
        with open(seqDir+record.id+".fasta", 'w') as f:
            f.write(">"+record.description+"\n")
            f.write(str(record.seq))

    print("input splitted")


@ray.remote
def blast(blast_in, blast_out, database, e_val= 1e-10, qcov=60, iterations=3 ):
    #build command line
    output_format = 7 #format tableau
    #e_val = 1e-10 #false positive probability
    #qcov = 60 #percentage overlap
    #iterations = 3 
    cline = "psiblast " + '-db '+ database + ' -query ' + blast_in + ' -evalue ' + str(e_val) + ' -out ' + blast_out + ' -num_iterations '+ str(iterations) + ' -outfmt ' + str(output_format) +' -qcov_hsp_perc ' + str(qcov)
    #execute
    subprocess.run(list(str(cline).split(' ')))


#@ray.remote #paralleliser pour optimisation
def find_homolog(seqDir, outDir, database, e_val= 1e-10, qcov=60, iterations=3):
    ## execute blast commandline on every query
    #command example: psiblast -db local_db/swissprot -query "$file" -num_iterations 3 -evalue 1e-10 -qcov_hsp_perc 60 -out outputs/"$file" -outfmt 7

    if not os.path.exists(outDir):
        os.mkdir(outDir)

    results_ids = []
    for seqfile in os.listdir(seqDir):
        blast_in = os.path.join(seqDir, seqfile)
        blast_out = outDir + blast_in.split("/")[-1].split('.')[0]+'.out'
        res_id = blast.remote(blast_in, blast_out, database, e_val= e_val, qcov=qcov, iterations=iterations)
        results_ids.append(res_id)

    ray.get(results_ids)


#transform blast output
@ray.remote
def fetch_seq(dictionnaire, dirName, fileName, database_seqs):
    #blast output(tableau) -> multifasta
    subjects = dictionnaire["infos"]['subjects']
    #print(subjects)
    outFile = os.path.join(dirName, fileName)
    with open(outFile, 'w') as out:
        for record in SeqIO.parse(database_seqs, 'fasta'):
            subject_id =  record.description.split('|')[1]
            if(subject_id in subjects):
                out.write(">"+subject_id+"\n")
                out.write(str(record.seq)+"\n")

    return outFile


#alignement muscle 
#input [fichier à aligner, multifasta] [fichier de sortie, alignement]
@ray.remote
def alignM( raw, aligned ):     #Muscle
    #wrap command
    cline = 'muscle'+" -in " + raw +" -out " + aligned
    # cline = MuscleCommandline(input= raw, out= aligned)
    print(cline)
    #run command
    subprocess.run(list(str(cline).split(' ')))
        

@ray.remote
def crop(records, indices, outFile, seq_id):
    with open(outFile,'w') as f:
        for record in records:
            f.write(">"+str(record.id)+'\n')
            sequence = str(record.seq)
            decal = 0 
            for ind in indices:
                ind -= decal
                sequence = sequence[:ind] + sequence[ind+1:]
                decal += 1
            f.write(sequence + '\n\n')


#parse the blast output to retrieve homologuous sequences
def sequences_from_output(outDir, homDir, database_seqs):
    if not os.path.exists(homDir):
        os.mkdir(homDir)

    results_ids = []
    for outfile in os.listdir(outDir):  # for each blast output
        blast_out = os.path.join(outDir, outfile)
        dictionnaire, convergence_status = parse(blast_out) # parse output matrix to get sequence IDs

        seqName = outfile[:-4] + '.fasta'
        res_id = fetch_seq.remote(dictionnaire, homDir, seqName, database_seqs) # get homologous sequences from IDs
        results_ids.append(res_id)

    ray.get(results_ids)

#align all found homologs
def align_homologs(aliDir,homDir):
    if not os.path.exists(aliDir):
        os.mkdir(aliDir)
    
    results_ids = []
    for seqName in os.listdir(homDir):  
        homFile = os.path.join(homDir, seqName)
        aliFile = os.path.join(aliDir, seqName) #same name different directory
        res_id = alignM.remote( homFile, aliFile)   # align homologous sequences
        results_ids.append(res_id)
    ray.get(results_ids)
    

#remove gaps position from query sequence in alignment, and necessarly remove the same positions in all other aligned sequences
def crop_ali(outDir, aliDir, cropDir, ssDir):
    if not os.path.exists(cropDir):
        os.mkdir(cropDir)

    results_ids = []
    for seqName in os.listdir(aliDir):
        ali_out = os.path.join(aliDir, seqName)

        with open(os.path.join(ssDir, seqName), 'r') as f:
            line = next(f).split('|')[1]
            sequence_ID = line.strip()

        try: 
            handler = open(ali_out)
            alignment = AlignIO.read(handler, "fasta")
            size = alignment.get_alignment_length()
            handler.close()

            records = []
            delete_ind = []
            for record in alignment :
                records.append( record )
                if(record.id in sequence_ID):   #c'est la sequence query
                    for i in range(size):
                        sequence = str(record.seq)
                        if(sequence[i] == "-"):
                            delete_ind.append(i)

            # print(records)
            # print(delete_ind)
            crop_ali_out = os.path.join(cropDir, seqName)
            res_id = crop.remote(records, delete_ind, crop_ali_out, sequence_ID)
            results_ids.append(res_id)

        except FileNotFoundError:
            print("NOT FOUND: ",ali_out)
            continue
        except ValueError:
            print("CORRUPTED: ",ali_out)

    ray.get(results_ids)


@ray.remote
def read_hom(homFile):
    seq_length = []
    with open(homFile,'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq = record.seq
            seq_length.append(len(seq))
    seq_length = np.array(seq_length)
    return seq_length


#@ray.remote
def score_blast(seqDir, scoreDir, database, database_seqs, e_val, qcov, iterations):
    epoch_outDir = os.path.join(scoreDir, 'out-'+str(qcov)) #identify epoch blast out
    epoch_homDir = os.path.join(scoreDir, 'hom-'+str(qcov)) #identify epoch homologs

    if not os.path.exists(epoch_outDir):
        os.mkdir(epoch_outDir)
    if not os.path.exists(epoch_homDir):
        os.mkdir(epoch_homDir)

    find_homolog(seqDir, epoch_outDir, database, e_val, qcov, iterations)
    sequences_from_output(epoch_outDir, epoch_homDir, database_seqs)

    object_ids = []
    for seqName in os.listdir(epoch_homDir):  
        #print(seqName)
        homFile = os.path.join(epoch_homDir, seqName)
        obj_id = read_hom.remote(homFile)
        object_ids.append(obj_id)

    all_seq_length = ray.get(object_ids) 
    nb_seq = sum([len(x) for x in all_seq_length])
    seq_var = []
    for ll in all_seq_length:
        seq_var.append( np.sqrt(np.var(ll)) ) #ecart type
    seq_var = np.array(seq_var)
    mean_var = np.mean(seq_var)

    score = nb_seq/mean_var
    return (qcov, score)


def optimize_cov(seqDir, scoreDir, database, database_seqs):
    if not os.path.exists(scoreDir):
        os.mkdir(scoreDir)

    iterations=3
    e_val= 1e-10

    results_ids = []
    for qcov in range(40,90,1):        
        res_id = score_blast(seqDir, scoreDir, database, database_seqs, e_val, qcov, iterations)
        results_ids.append(res_id)
    qcov_perf = ray.get(results_ids)


    with open('qcov_perf.pickle', 'wb') as handle:
        pkl.dump(qcov_perf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    best_param = max(qcov_perf,key=lambda item:item[1])
    print(best_param)
    with open('best_param.txt','w') as f:
        f.write('qcov: '+str(best_param[0])+'  score: '+str(best_param[1]))



# local
database_seqs = "/home/alex/Documents/data/uniprot_sprot.fasta"
database_blast = "/home/alex/Documents/data/local_db/swissprot" 

# # cluster #TODO remember to set back to these before running jobs
# database_seqs = "/shared/bank/uniprot_swissprot/current/fasta/uniprot_swissprot.fsa"
# database_blast = "/shared/bank/uniprot_swissprot/current/blast/" 

pre = "" #"blast_paralized/" #TODO
queries = pre+"input.fasta"
seqDir = pre+"single_seq/"

results_dir = pre+"results"
outDir = results_dir+"/outputs/"
homDir = results_dir+"/homologs/"
aliDir = results_dir+"/alignements/"
cropDir = results_dir+"/cropped_alignements/"

# initialize(seqDir, results_dir, queries) 
# find_homolog(seq, outDir, database_blast )
# sequences_from_output(outDir, homDir, database_seqs)
# align_homologs(aliDir, homDir)
# crop_ali(outDir, aliDir, cropDir, seqDir)


scoreDir = results_dir+"/blast_scores/"
optimize_cov(seqDir, scoreDir, database_blast, database_seqs)

print("finished")


# liste = []
# for outfile in os.listdir("results/outputs/"):  # for each blast output
#     blast_out = os.path.join("results/outputs/", outfile)
#     dictionnaire, conv = parse(blast_out) # parse output matrix to get sequence IDs
#     liste.append(conv)

# print("blast convergence rate: ", sum(liste)/len(liste))
