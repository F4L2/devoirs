import numpy as np
import matplotlib.pyplot as plt
import os 
import imageio
import gc

def color_split( X, k=3):
    N = len(X)
    mult = 1
    # x_classed = [[]] * k 
    x_classed = []
    cur_classe = []

    rgb = np.array([1.0, 0.5, 0.0])
    c_step = 1/k
    colors = []

    for i in range(N):
        if( i >= mult*N/k ):
            x_classed.append(cur_classe)
            cur_classe = []
            mult += 1
            rgb -= c_step
            for c_ind in range(3):
                if(rgb[c_ind] < 0):
                    rgb[c_ind] += 1

        if(i >= (mult-1)*N/k and i < mult*N/k):
            cur_classe.append(X[i])
            colors.append(list(rgb))
    x_classed.append(cur_classe)
    return x_classed, colors


def make_movie(screen_path, movie_name= 'movie.gif'):
    files = []
    imgs = []
    for r, d, f in os.walk(screen_path):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))
    #chronological order
    files = sorted(files)
    for f in files:
        imgs.append( imageio.imread(f) )
        gc.collect()
    imageio.mimsave(movie_name, imgs)
    print(movie_name)

# make_movie("savfig")  #/!\ Recommended RAM >4go 