'''fetch statistics from results folder and use the data'''

import os

out = open("TABLEAU3_2.txt",'w')

dict_mut = {}

for dirname in os.listdir('results'):
    sim_stat_file = os.path.join('results', dirname,'statistics.txt')
    out.write("---TABLEAU "+dirname+' ---------------------------------\n')
    with open(sim_stat_file, 'r') as f:
        for line in f:
            if("Statistics" in line):
                break
        for line in f:
            out.write(line)
            items = line.split(":")
            items[1] = items[1].strip()
            
            if(items[0] not in dict_mut):
                dict_mut[items[0]] = int(items[1])
            else:
                dict_mut[items[0]] += int(items[1])
        out.write('\n\n')

out.write("_________________")
out.write("\n---OVERALL-------\n")

for k,v in dict_mut.items():
    if(k == 'WGD'):
        out.write(k+":\t\t"+str(v)+"\n")
    else:
        out.write(k+":\t"+str(v)+"\n")

maximum = max(dict_mut, key=dict_mut.get)
minimum = min(dict_mut, key=dict_mut.get)


out.write("\n------------\n")
out.write("Most :\t" + maximum + " | " + str(dict_mut[maximum]) )
out.write('\n')
out.write("Least :\t" + minimum + " | " + str(dict_mut[minimum]) )