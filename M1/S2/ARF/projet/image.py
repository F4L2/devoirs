import matplotlib.colors as plc 
import matplotlib.pyplot as plt
import random
import numpy as np


def open_in_hsv(image):
    im=plt.imread(image)[ :, :, :3] #on garde que les 3 premieres composantes, la transparence est inutile
    return plc.rgb_to_hsv(im)

def hsv_to_pic(pixels):
    pixels = plc.hsv_to_rgb(pixels)
    plt.imshow(pixels) #afficher l’image 
    plt.show()

def compare_pic( original, noised, repaired ):
    repaired = plc.hsv_to_rgb(repaired)
    noised = plc.hsv_to_rgb(noised)
    original = plc.hsv_to_rgb(original)
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(original)
    axarr[1].imshow(noised)
    axarr[2].imshow(repaired)
    plt.savefig("last_try.png")
    plt.show()

def patch_visual( patch_mat ):
    rows = patch_mat.shape[0]
    cols = patch_mat.shape[1]
    f, axarr = plt.subplots(rows,cols)

    for i in range(len(patch_mat)):
        for j in range(len(patch_mat[i])):
            patch = patch_mat[i][j]
            img = plc.hsv_to_rgb(patch)
            axarr[i,j].imshow(img)
    plt.savefig("patch_matrix.png")
    plt.show()
    

def noise(pixels, perc):
    new_pixels = pixels.copy()
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            if(random.random() < perc):
                new_pixels[i][j][0]=-100
                new_pixels[i][j][1]=-100
                new_pixels[i][j][2]=-100
    return new_pixels


def clean_format(pixels):
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            for k in range(3):
                if(pixels[i][j][k] < -1 ):
                    pixels[i][j][k]=0
                else : 
                    if( pixels[i][j][k] >1):
                        pixels[i][j][k]=1
                    if( pixels[i][j][k] < 0):
                        pixels[i][j][k] = -1 * pixels[i][j][k] 
    return pixels



def delete_rect(original_pixels,i,j,height,width):
    pixels = original_pixels.copy()
    if( i + height > len(pixels)):
        print("height exceeds image")
    if( j + width > len(pixels[i])):
        print("width exceeds image")
    
    for c in range(i,i+height):
        for l in range(j,j+width):
            pixels[c][l][0] = -100
            pixels[c][l][1] = -100
            pixels[c][l][2] = -100

    return pixels
        

def get_patch(pixels,i,j,h):
    patch = []
    for c in range(i-int(h/2),i+int(h/2)):
        if(c<0 or c>len(pixels)):
            continue
        line = []
        for l in range(j-int(h/2),j+int(h/2)):
            if(l<0 or l>len(pixels[c])):
                continue
            line.append(pixels[c][l])
        line = np.array(line)
        patch.append(line)
    return np.array(patch)


def scan_patch(patch):
    for line in patch:
        if( -100 in line[:]):
            return True
    return False

def scan_vec(vec):
    if( -100 in vec ):
        return True
    return False


def classify( vec ):
    #input signal
    mu = vec.mean()
    sig = vec.var()

    signal = np.random.normal(mu, sig)


    #process signal bounds, [-1,1]
    if(signal < -1):
        signal = -1
    elif(signal > 1):
        signal = 1

    # signal = np.sign(signal)

    return signal


def split_patches(patch_mat):
    #returns noisy and clean patches
    noisy_patch = []
    clean_patch = []
    for row in patch_mat:
        for patch in row:
            if(scan_patch(patch)):
                noisy_patch.append(patch)
            else:
                clean_patch.append(patch)

    return np.array(noisy_patch), np.array(clean_patch)

def split_vecs( vec_list ):
    noisy_vec = []
    clean_vec = []
    for vec in vec_list:
        if(scan_vec(vec)):
            noisy_vec.append(vec)
        else:
            clean_vec.append(vec)
    return np.array(noisy_vec), np.array(clean_vec)

def find_patch(original_pixels,h):
    pixels = original_pixels.copy()
    patch_list = []
    for i in range( int(h/2),len(pixels)-int(h/2)+1, h):
        row = []
        for j in range( int(h/2),len(pixels[i])-int(h/2)+1, h):
            row.append(get_patch(pixels,i,j,h))
        patch_list.append(row)
    return np.array(patch_list)

def clean_vec (vec):
    return np.where(vec == -100, 0, vec)


def vectorize(patch):
    shape = patch.shape
    i = shape[0]
    j = shape[1]
    hsv = shape[2]
    # new_shape = (i*j, hsv)
    new_shape = (i*j * hsv)
    return patch.reshape( new_shape ) , shape



def vectorize_list( patch_liste ):
    vec_liste = []
    for patch in patch_liste:
        vec, patch_shape = vectorize(patch)
        vec_liste.append(vec)
    return np.array(vec_liste), patch_shape

def vectorize_part_mat( patch_mat ):
    vec_liste = []
    for row in patch_mat:
        for patch in row:
            vec, patch_shape = vectorize(patch)
            vec_liste.append(vec)
    return np.array(vec_liste)

def vectorize_mat( patch_mat ):
    mat_shape = patch_mat.shape
    new_shape = (mat_shape[0]*mat_shape[1], mat_shape[2], mat_shape[3], mat_shape[4])
    return patch_mat.reshape(new_shape), mat_shape

def classify_liste( liste_vec ):
    vectors = []
    for vec in liste_vec:
        classe = classify( vec )
        vectors.append(classe)
    return np.array(vectors)


def patchify(vec, shape):
    return vec.reshape(shape)



def reconstruct( patches, shape ):
    patch_mat = patches.reshape(shape)

    img_matrix = []
    for row in patch_mat:
        img_row = row[0]
        for j in range(1,len(row)):
            col = row[j]
            img_row = np.hstack( (img_row, col) )
        img_matrix.append(img_row)
    
    img = img_matrix[0]
    for i in range(1, len(img_matrix)):
        img = np.vstack( (img, img_matrix[i]) )

    return img


def find_corruption( vec_list ):
    indexes = []
    for ind in range(len(vec_list)):
        vec = vec_list[ind]
        if(scan_vec(vec)):
            indexes.append(ind)
    return indexes

def noise_in_patch( patch ):
    noisy_pixels = 0
    for row in patch:
        for pix in row:
            if(-100 in pix):
                noisy_pixels += 3
    return noisy_pixels / patch.size



def split_pixels( patch ):
    noisy_pixels = []
    dic = []
    for i in range(len(patch)):
        for j in range(len(patch[i])):
            pix = patch[i][j]
            if(-100 in pix):
                noisy_pixels.append( (i,j) )
            else:
                dic.append(pix)
    dic = np.array(dic)
    return noisy_pixels, dic


def filter_dic(patch, noisy_pix):
    dic = []
    for i in range(len(patch)):
        for j in range(len(patch[i])):
            if( (i,j) in noisy_pix ):
                continue
            pix = patch[i][j]
            dic.append(pix)
    dic = np.array(dic)
    return dic


def vectorize_pixel ( liste_pix ):
    shape = liste_pix.shape
    return liste_pix.reshape((shape[0]*shape[1])), shape

def pixelise( liste, shape ):
    return liste.reshape( shape )
