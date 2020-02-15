import os
import subprocess
from Bio import SeqIO

### génère les fichiers de séquences de clade qu'il faudra aligner 
LSC=[]#liste Clade
for temp in os.listdir("clades"):
    LSC.append(temp)

LC=[]#Liste Sous Clade
for temp in os.listdir("clades/"+LSC[0]):
    LC.append(temp)
    
    
    


for i in range(len(LC)-1):
    for j in range(i+1,len(LC)):
        d=[] #liste ressensant les  multiples allignements
        for l in LSC:
            #path de leacture pour les 2 clades d'une famille l
            path1="clades/"+l+"/"+LC[j]
            path2="clades/"+l+"/"+LC[i]
            PT="temp/"+LC[i]+LC[j] #path Temp
            
            #creation d'un dossier pour C1,C2
            if os.path.isdir(PT)==False:
                os.mkdir(PT)
                
            #écriture de la concaténation de C1,C2 pour une famille donnée dans le folder fait pour
            remove=False
            with open(PT+"/"+l, 'w') as outfile:                
                if (os.path.isfile(path1)==True and os.path.isfile(path2)==True):                    
                    for fname in [path1,path2]:
                        with open(fname) as infile:
                            for line in infile:
                                outfile.write(line)      
                else :
                    remove=True


            if remove==True:
                os.remove(PT+"/"+l)
            
            
            #allignements des file:
            if os.path.isfile(PT+"/"+l)==True:
                cline = './ok -i '+ PT+"/"+l + ' -o '+ PT+"/"+l+"_ali" +' --force' #+" --outfmt=phy" 
                subprocess.run(list(str(cline).split(' ')))  
                d.append(PT+"/"+l+"_ali")
        
        #concatenation des files    
        dic = {}
        for ali_file in d:
            print(ali_file)
            for record in SeqIO.parse(ali_file, 'fasta'):
                seq_id, specy = record.description.strip().split("_")
                if specy not in dic:
                    dic[specy] = str(record.seq) 
                else:
                    dic[specy] += str(record.seq)

        with open(PT+"/"+"aaasuperalign.fasta", 'w') as out:
            for k,v in dic.items():
                out.write(">"+k+'\n')
                out.write(v + '\n')

#convert superalign to phylip format

            
            
                
                
            
                
                
        print(LC[i],LC[j]," traité")
            














