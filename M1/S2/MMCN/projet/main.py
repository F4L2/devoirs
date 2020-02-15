from junk import os, np, plt

# 'act_max' / 'dist_min'
# method = 'act_max'
method = 'dist_min' 

'''init'''
N = 1000
K = 25
colors = np.random.rand(K,3)
X = np.random.multivariate_normal( (0,0), np.eye(2)*10, N)

#initialiser le vecteur poids 2D aléatoirement entre [-1,1]
W = np.random.uniform( -1, 1, (K,2) )

T = 100000
eta = 0.01

'''learning'''
if not os.path.isdir("savfig/") :
    os.mkdir("savfig/")


# initialisation graphique
plt.figure( figsize= (10,5.5) ); plt.clf()

for i in range(T):    
    ind = np.random.randint(N)
    x = X[ind]
    y = np.dot(W, x)

    '''apprentissage compétitif'''
    if( method == 'act_max' ): 
        #y le plus grand
        i_max = np.argmax(y)
        W[i_max] += eta*(x - W[i_max])

    elif( method == 'dist_min' ):
        #vecteur poids le plus proche
        distances = np.array([ np.linalg.norm( weight - x ) for weight in W ])
        j_min = np.argmin(distances)
        W[j_min] += eta*(x - W[j_min])

    if i%100==0:
        print('Iteration %d' % i)                          

        plt.subplot(121); plt.cla()

        plt.scatter( X[:,0], X[:,1], c= [(0.7, 0.7, 0.7)]* N)    #all
        plt.scatter( x[0], x[1], c=[(0.3,0.1,0.9)] )             #selected
        for k in range(K):
            plt.plot([0, W[k,0]], [0, W[k,1]], '-g', lw=3)

        plt.axhline(0, ls=':', color='k')                    
        plt.axvline(0, ls=':', color='k')                    
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.axis('scaled')                                 
        plt.draw()

        # plt.savefig("savfig/instance"+str(i)+".png")  #pour faire le gif.
        plt.pause(0.1)


'''classif'''
classes = []

if( method == 'act_max' ): 
    # y_max
    for x in X:
        y = np.dot(W,x)
        win = np.argmax(y)
        classes.append(colors[win])

    plt.subplot(122); plt.cla()
    plt.scatter( X[:,0], X[:,1], color=classes )  
    plt.axhline(0, ls=':', color='k')                    
    plt.axvline(0, ls=':', color='k')                    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis('scaled')                                    
    plt.draw()

    plt.savefig("act_max.png")
    plt.show()

elif( method == 'dist_min' ):
    # dist_min
    for x in X:
        distances = np.array([ np.linalg.norm( weight - x ) for weight in W ])
        win = np.argmin(distances)
        classes.append(colors[win])

    plt.subplot(122); plt.cla()
    plt.scatter( X[:,0], X[:,1], color=classes )  
    plt.axhline(0, ls=':', color='k')                    
    plt.axvline(0, ls=':', color='k')                    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis('scaled')                                    
    plt.draw()

    plt.savefig("dist_min.png")
    plt.show()