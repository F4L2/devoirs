from inpainting import * 

'''Question 3'''

reg = Lasso(max_iter=10000 ,alpha= 0.005)
h = 8
noise_center = (40,20)

pixels = open_in_hsv("rsc/sample.png")
n_pixels = delete_rect( pixels, noise_center[0], noise_center[1], 16, 8)
new_pixels = n_pixels.copy()

done = False
it = 0
while not done:
    patch_mat = find_patch(new_pixels, h)
    all_patches, mat_shape = vectorize_mat(patch_mat)

    noisy, clean = split_patches(patch_mat)
    print(noisy.shape, clean.shape)
    if( len(clean) == 0):
        print("image too corrupted to reconstruct")
        break

    corrupt_index = find_corruption(all_patches)
    if(len(corrupt_index) == 0):
        done = True  
        break

    noisy_pixel, dictionnaire = split_pixels( new_pixels )
    priority_liste = priority(noisy_pixel, noise_center)

    #choose priority
    # priority_liste = priority_liste[::-1] #inverse la priorité pour prioritiser le centre de la partie manquante
    top_priority_point = priority_liste[0]

    print(top_priority_point)
    i,j = top_priority_point


    # for i,j in priority_liste: #take max priority
    patch = get_patch(new_pixels, i, j, h)
    vec, patch_shape = vectorize(patch)

    noisy_pixel, not_noisy_pixel = split_pixels( patch )
    noisy_pix, not_noisy_pixel = split_pixels(patch)
    if(len(not_noisy_pixel) == 0):    #pas de dictionnaire, un patch entièrement corrompu
        continue
    not_noisy_pixel, pixel_shape = vectorize_pixel(not_noisy_pixel)

    tmp_patch = clean
    dic = []
    training_dic = []
    for p in tmp_patch:
        full_vec, devnull = vectorize(p)
        dic.append(full_vec)

        d = filter_dic(p, noisy_pixel) #sélectionner seulement les pixel non corrompu du patch corrompu
        d, devnull = vectorize_pixel(d)
        training_dic.append(d)
    dic = np.array(dic) #full vector with all pixels all patch

    training_dic = np.array(training_dic) #filtered vector with only pixels expressed in the patch
    y, devnull = vectorize_pixel(filter_dic(patch, noisy_pixel)) #y = le patch choisi
    y = np.array(y)

    dic_weight = find_dic_weight(training_dic, y, reg)

    new_vec = vec
    for pix in range(len(vec)):
        if(vec[pix] != -100):                     # le pixel n'est pas corrompu, rien à faire
            continue
        new_vec[pix] = sum( dic[:,pix] * dic_weight[:] )
    new_patch = patchify(new_vec, patch_shape)

    #actualiser l'image avec le nouveau patch
    p_i = 0
    p_j = 0
    for c in range(i-int(h/2),i+int(h/2)):
        p_j = 0
        for l in range(j-int(h/2),j+int(h/2)):
            new_pixels[c][l] = new_patch[p_i][p_j] 
            p_j += 1
        p_i += 1

    it+=1
    print(it)
    
new_pixels = reconstruct( all_patches, mat_shape )
new_pixels = clean_format(new_pixels)
n_pixels = clean_format(n_pixels)

compare_pic( pixels, n_pixels, new_pixels)

