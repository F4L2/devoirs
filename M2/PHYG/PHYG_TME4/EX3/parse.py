from Bio import SeqIO
import os
import subprocess


species_list = []
with open("species.list", 'r') as f:
    for line in f:
        species_list.append(line.strip())

out_dir = "selected_sequences"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for sfile in os.listdir("TME4_sequences"):
    pfam_file = os.path.join("TME4_sequences", sfile)
    out_file = os.path.join(out_dir, sfile)
    
    fam_species_list = species_list.copy()
    with open(out_file, 'w') as out:
        for record in SeqIO.parse(pfam_file, 'fasta'):
            seq_id, specy = record.description.strip().split("_")
            if specy in fam_species_list:
                out.write(">"+record.description+'\n')
                out.write(str(record.seq) + '\n')
                fam_species_list.remove(specy)

    if(len(fam_species_list) > 0): #one specy not found
        os.remove(out_file)


ali_dir = "alignments"
if not os.path.exists(ali_dir):
    os.mkdir(ali_dir)

for pfile in os.listdir(out_dir):
    pfam_file = os.path.join(out_dir, pfile)
    ali_file = os.path.join(ali_dir, pfile)

    cline = './clustalo -i '+ pfam_file + ' -o '+ ali_file #+' --outfmt=phy'
    subprocess.run(list(str(cline).split(' ')))



#concat 

dic = {}
for afile in os.listdir(ali_dir):
    ali_file = os.path.join(ali_dir, afile)

    for record in SeqIO.parse(ali_file, 'fasta'):
        seq_id, specy = record.description.strip().split("_")
        if specy not in dic:
            dic[specy] = str(record.seq) 
        else:
            dic[specy] += str(record.seq)

with open("superalign.fasta", 'w') as out:
    for k,v in dic.items():
        out.write(">"+k+'\n')
        out.write(v + '\n')
#convert superalign to phylip format