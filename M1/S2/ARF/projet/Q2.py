from inpainting import * 

'''Question 2'''

pixels = open_in_hsv("rsc/sample.png")
n_pixels = noise(pixels, 0.1)   #combine with delete rect()

patch_size = 8
patch_mat = find_patch(n_pixels, patch_size)
all_patches, mat_shape = vectorize_mat(patch_mat)
all_vecs, patch_shape = vectorize_list(all_patches)

# patch_visual( patch_mat )

#tricher en prenant des patch de l'image d'origine
cheat_mat = find_patch(pixels, patch_size)
cheat_patch, mat_shape = vectorize_mat(cheat_mat)
cheat_vec, patch_shape = vectorize_list(cheat_patch)

noisy, clean = split_patches(patch_mat)
print(noisy.shape, clean.shape)

corrupt_index = find_corruption(all_vecs)
reg = Lasso(max_iter=10000 ,alpha= 0.005)

done = False
it = 0
while not done:
    #pour tous les patch corrompus
    print(corrupt_index)
    for i in corrupt_index:
        patch = all_patches[i]
        vec, devnull = vectorize(patch)

        # tmp_patch = list(clean)                   #sans triche, à faire que s'il y a peu de bruit ou bruit localisé

        ## on triche en prenant les patchs de l'image d'origine 
        tmp_patch = list(cheat_patch.copy())
        tmp_patch.pop(i)                            #on enlève le patch sélectionné du training_dic, on triche mais quand même! 
        tmp_patch = np.array(tmp_patch) 

        # ne pas prendre en compte les pixels corrompus dans le patch pour le training_dic
        noisy_pixel, not_noisy_pixel = split_pixels( patch )
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
        print(i, dic_weight)


        new_vec = vec
        for j in range(len(vec)):
            if(vec[j] != -100):                     # le pixel n'est pas corrompu, rien à faire
                continue
            new_vec[j] = sum( dic[:,j] * dic_weight[:] )

        new_patch = patchify(new_vec, patch_shape)
        all_patches[i] = new_patch

    # mis à jour des index  
    all_vecs, patch_shape = vectorize_list(all_patches)
    corrupt_index = find_corruption(all_vecs)
    if(len(corrupt_index) == 0):
        done = True    
    n_vecs, c_vecs = split_vecs(all_vecs)

    it+=1
    print(it)

new_pixels = reconstruct( all_patches, mat_shape )

new_pixels = clean_format(new_pixels)
n_pixels = clean_format(n_pixels)

compare_pic( pixels, n_pixels, new_pixels)



#Q2
# augmenter alpha réduit le nombre de solution, on obtient plus fréquemment des vecteurs poids nuls 
# en baissant alpha on obtient plus facilement des vecteurs poids non-nuls, non seulement ça mais aussi plus de poids non-nuls en général.
# avec un alpha trop grand on n'obtient pas de solution, mais avec un alpha très petit, la solution n'est aussi pas très expressif

