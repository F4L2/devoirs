from Bio import SeqIO


species_list = []
with open("species.list", 'r') as f:
    for line in f:
        species_list.append(line.strip())

with open("selected_sequences.fasta", 'w') as out:
    for record in SeqIO.parse("PF01599.fasta", 'fasta'):
        seq_id, specy = record.description.strip().split("_")
        if specy in species_list:
            out.write(">"+record.description+'\n')
            out.write(str(record.seq) + '\n')
            species_list.remove(specy)
