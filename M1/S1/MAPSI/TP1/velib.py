import requests
import pickle as pkl
import time

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')   


fname = "dataVelib.pkl"
f= open(fname,'rb')
data = pkl.load(f, encoding='latin1') # option encoding pour le passage py2 => py3
f.close()

'''
url = "https://opendata.paris.fr/explore/embed/dataset/velib-disponibilite-en-temps-reel/table/?dataChart=eyJxdWVyaWVzIjpbeyJjb25maWciOnsiZGF0YXNldCI6InZlbGliLWRpc3BvbmliaWxpdGUtZW4tdGVtcHMtcmVlbCIsIm9wdGlvbnMiOnt9fSwiY2hhcnRzIjpbeyJhbGlnbk1vbnRoIjp0cnVlLCJ0eXBlIjoiY29sdW1uIiwiZnVuYyI6IkFWRyIsInlBeGlzIjoic3RhdGlvbl9pZCIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6IiMyNjM4OTIifV0sInhBeGlzIjoic3RhdGlvbl9pZCIsIm1heHBvaW50cyI6NTAsInNvcnQiOiIifV0sInRpbWVzY2FsZSI6IiIsImRpc3BsYXlMZWdlbmQiOnRydWUsImFsaWduTW9udGgiOnRydWV9"
dataStation = requests.get(url)
data = dataStation.json()

#enrichissement avec api google map
urlGoogleAPI = "https://maps.googleapis.com/maps/api/elevation/json?locations="

for s in data:
    position = "%f,%f"%(s['position']['lat'],s['position']['lng'])
    alt = requests.get(urlGoogleAPI+position)
    assert(alt.json()['status'] == "OK") # vérification de la réussite
    s[u'alt'] = alt.json()['results'][0]['elevation'] # enrichissement
    time.sleep(0.1) # pour ne pas se faire bannir par Google
'''

listeARR = []
listePdis = []
listePtot = []

for station in data:
    nbVeloDispo = station['available_bikes']
    nbVeloTotal = station['bike_stands']
    arrondissement = station['number'] / 1000
    #print(arrondissement)
    if arrondissement >= 1 and arrondissement <= 20 : 
        listeARR.append(arrondissement)
        listePdis.append(nbVeloDispo)
        listePtot.append(nbVeloTotal)

matrice = np.vstack( (listeARR, listePdis, listePtot) )  
matrice.transpose()
print(matrice)

f= open('coordVelib.pkl','wb')
pkl.dump(data,f) # penser à sauver les données pour éviter de refaire les opérations
f.close()
