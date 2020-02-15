import numpy as np

def tabcopi(t):
    t2=[]
    for i in t:
        t2.append(i)
    return t2

seq1 = "CATGAC"
seq2 = "TCTGAAC"

prot1 = ("MGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQ"
        "TKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSF"
        "LVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELV"
        "HHHSTVADGLITTLHYPAP")

prot2 = ("GAMDPSEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFV"
        "ALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGW"
        "VPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESE"
        "SSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHST"
        "VADGLITTLHYPAPKRNKPTIYGVSPNYDKWEMERTDITMKHKLGGG"
        "QYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNL"
        "VQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVSAVVLLYMATQIS"
        "SAMEYLEKKNFIHRNLAARNCLVGENHLVKVADFGLSRLMTGDTYTAH"
        "AGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDL"
        "SQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAF"
        "ETMFQESSISDEVEKELGKRGT")

g = -11
g2 = -1
mismatch = -2
match = 2

def isMatch (a,b):
    return a==b
    
def NW (A, B):
    mat = np.zeros( (len(A)+1, len(B)+1) ) #[numeroligner][numerocolone]
    matpass = [] #(-1,-1),(-1,0),(0,-1)
    p=[False,False,False]
    #init matpass
    for i in range(1+len(A)):
        matpass.append(tabcopi([p]*(1+len(B))))
    
    #init mat
    mat[1][0] = g
    matpass[1][0]=[False,True,False]
    for i in range( 2,1+len(A)):
        mat[i][0] = (i-1 )* g2 +g
        matpass[i][0]=[False,True,False]
    mat[0][1] = g
    matpass[0][1]=[False,False,True]
    for i in range(1, 1+len(B)):
        mat[0][i] = (i-1 )* g2 +g
        matpass[0][i]=[False,False,True]
    
    #récurrence
    for i in range( 1, 1+len(A) ):
        for j in range( 1, 1+len(B) ):
            val=[]
            if A[i-1]== B[j-1]:
                val.append(mat[i-1][j-1]+match)
            else :
                val.append(mat[i-1][j-1]+mismatch)
            if matpass[i-1][j][1] :
                val.append(mat[i-1][j]+g2)
            else:
                val.append(mat[i-1][j]+g)
            if matpass[i][j-1][1] :
                val.append(mat[i][j-1]+g2)
            else:
                val.append(mat[i][j-1]+g)
            mat[i][j]= max(val)
            matpass[i][j]=[val[0]==max(val),val[1]==max(val),val[2]==max(val)]
    
    #traceback
    A2=""
    B2=""
    i = len(A)
    j = len(B)
    minimal = (mismatch+g)*(len(A)+len(B)+10) #unexpected low value
    while i > 0 or j > 0:
            val=[minimal]*3
            if matpass[i][j][0] == True:
                val[0]=mat[i-1][j-1]
            if matpass[i][j][1] == True:
                val[1]=mat[i-1][j]
            if matpass[i][j][2] == True:
                val[2]=mat[i][j-1]
                
            if max(val)==mat[i-1][j-1]:
                A2+=A[i-1]
                B2+=B[j-1]
                i-=1
                j-=1
            elif max(val)==mat[i-1][j]:
                A2+=A[i-1]
                B2+='-'
                i-=1
            elif max(val)==mat[i][j-1]:
                A2+='-'
                B2+=B[j-1]
                j-=1
            else:
                print("erreur")
    print(mat)
    A3=""
    B3=""
    for i in range(len(A2)):
        A3+=A2[len(A2)-1-i]
        B3+=B2[len(A2)-1-i]
    
    print(A3)
    print(B3)
    

def SW (A, B):
    mat = np.zeros( (len(A)+1, len(B)+1) ) #[numeroligner][numerocolone]
    matpass = [] #(-1,-1),(-1,0),(0,-1)
    p=[False,False,False]
    #init matpass
    for i in range(1+len(A)):
        matpass.append(tabcopi([p]*(1+len(B))))
    
    #init mat

    for i in range( 2,1+len(A)):
        mat[i][0] = 0
        
    for i in range(1, 1+len(B)):
        mat[0][i] = 0
    
    #récurrence
    for i in range( 1, 1+len(A) ):
        for j in range( 1, 1+len(B) ):
            val=[]
            if A[i-1]== B[j-1]:
                val.append(mat[i-1][j-1]+match)
            else :
                val.append(mat[i-1][j-1]+mismatch)
            if matpass[i-1][j][1] :
                val.append(mat[i-1][j]+g2)
            else:
                val.append(mat[i-1][j]+g)
            if matpass[i][j-1][1] :
                val.append(mat[i][j-1]+g2)
            else:
                val.append(mat[i][j-1]+g)
            val.append(0)
            mat[i][j]= max(val)
            matpass[i][j]=[val[0]==max(val),val[1]==max(val),val[2]==max(val)]
    
    #traceback
    A2=""
    B2=""
    maxi=0
    for i1 in range( 1, 1+len(A) ):
        for j1 in range( 1, 1+len(B) ):
            if mat[i1][j1]>=maxi:
                maxi=mat[i1][j1]
                i=i1
                j=j1
    
    minimal = (mismatch+g+g2)*(len(A)+len(B)+10)
    test=True
    while test:
            val=[minimal]*3
            if matpass[i][j][0] == True:
                val[0]=mat[i-1][j-1]
            if matpass[i][j][1] == True:
                val[1]=mat[i-1][j]
            if matpass[i][j][2] == True:
                val[2]=mat[i][j-1]
                
            if max(val)==mat[i-1][j-1]:
                A2+=A[i-1]
                B2+=B[j-1]
                i-=1
                j-=1
            elif max(val)==mat[i-1][j]:
                A2+=A[i-1]
                B2+='-'
                i-=1
            elif max(val)==mat[i][j-1]:
                A2+='-'
                B2+=B[j-1]
                j-=1
            else:
                test=False
    print(mat)
    A3=""
    B3=""
    for i in range(len(A2)):
        A3+=A2[len(A2)-1-i]
        B3+=B2[len(A2)-1-i]
    
    print(A3)
    print(B3)


#######################


#NW(seq2, seq1)
#SW(seq2,seq1)

#print(prot1)
#print(prot2)
NW(prot1, prot2)
