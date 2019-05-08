import cv2 as cv
import numpy as np
import datetime
import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


WINDOW_SIZE = 50
NUM_PIXELS_BUILD = 387999 # cnt_p/83
NUM_PIXELS_NOT_BUILD = 491909 # cnt_np/87
FEATURES = ['contrast', 'homogeneity', 'energy']

# 32203917/3 p
# 42796083/3 np
# 32203917/387999 = 83
# 42796083/491909 = 87
# largura_altura(387999) (171, 2269)
# largura_altura(32203917) (4731, 6807)
# largura_altura(42796083) (6501, 6583)
# largura_altura(491909) (227, 2167)

def largura_altura(pixels_predios):
    alt = int(pixels_predios ** 0.5)

    while (pixels_predios % alt != 0) :
        alt -= 1
    
    return alt, pixels_predios // alt

def main():
    img = cv.imread("../data/vienna16.tif", -1)
    img_gt = cv.imread("../data/gt/vienna16.tif", -1)
    # img = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    a = datetime.datetime.now()
    height, width, channels = img.shape
    arr_p = []
    arr_np = []
    cnt_p = 0
    cnt_np = 0
    for y in range(height):
        for x in range(width):
            if(img_gt[y,x] == 255):
                arr_p.append(img[y, x, 0])
                arr_p.append(img[y, x, 1])
                arr_p.append(img[y, x, 2])
                cnt_p += 3
            else:
                arr_np.append(img[y, x, 0])
                arr_np.append(img[y, x, 1])
                arr_np.append(img[y, x, 2])
                cnt_np += 3

    alt_p, larg_p = largura_altura(NUM_PIXELS_BUILD)
    alt_np, larg_np = largura_altura(NUM_PIXELS_NOT_BUILD)
    
    n_build = []
    build = []

    for i in range(1,84):
        l = (i - 1)*NUM_PIXELS_BUILD
        r = i*NUM_PIXELS_BUILD
        aux_p = np.array(arr_p[l:r], dtype = np.uint8)
        aux_p = np.reshape(aux_p, (alt_p, larg_p))
        glcm_p = greycomatrix(aux_p, [1], [0], 256, symmetric=True, normed=True)
        for feat in FEATURES:
            build.append(greycoprops(glcm_p, feat)[0,0])    

    for i in range(1,88):
        l = (i - 1)*NUM_PIXELS_NOT_BUILD
        r = i*NUM_PIXELS_NOT_BUILD
        aux_np = np.array(arr_np[l:r], dtype = np.uint8)
        aux_np = np.reshape(aux_np, (alt_np, larg_np))
        glcm_np = greycomatrix(aux_np, [1], [0], 256, symmetric=True, normed=True)
        for feat in FEATURES:
            n_build.append(greycoprops(glcm_np, feat)[0,0])

    build = np.array(build)
    n_build = np.array(n_build)

    # np.random.shuffle(build)
    # np.random.shuffle(n_build)

    # build = normalize(build)
    # n_build = normalize(n_build)
    print(build.shape)
    print(n_build.shape)
    
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, 
        n_jobs=-1, n_neighbors=5, p=2, weights='uniform')
    X_train = np.append(build, n_build)
    X_train = np.reshape(X_train, (X_train.shape[0]//len(FEATURES), len(FEATURES)))
    print(X_train)
    print(X_train.shape)
    Y_train = np.zeros((170), dtype = np.int)
    Y_train[84:170] = 1
    knn.fit(X_train, Y_train)

    ################
    # IMAGEM FINAL #
    ################
    img = cv.imread("../data/vienna22.tif", -1)
    # img = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    img_gt = cv.imread("../data/gt/vienna22.tif", -1)

    feat = []
    cnt = 0
    print("entrou")
    for y in range(0, height, WINDOW_SIZE):
        for x in range(0, width, WINDOW_SIZE):
            # arr = img[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
            arr_b = img[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE, 0]
            arr_g = img[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE, 1]
            arr_r = img[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE, 2]
            # arr = (arr_b + arr_g + arr_r)//3
            # arr = arr.astype(np.uint8)
            glcm_b = greycomatrix(arr_b, [1], [0], 256, symmetric=True, normed=True)
            glcm_g = greycomatrix(arr_g, [1], [0], 256, symmetric=True, normed=True)
            glcm_r = greycomatrix(arr_r, [1], [0], 256, symmetric=True, normed=True)
            for f in FEATURES:
                b = (greycoprops(glcm_b, f)[0,0])
                g = (greycoprops(glcm_g, f)[0,0])
                r = (greycoprops(glcm_r, f)[0,0])
                feat.append((b+g+r)/3)
            cnt += 1
    
    print("saiu do for")
    feat = np.array(feat)
    feat = feat.reshape(cnt, len(FEATURES))
    # np.random.shuffle(feat)
    # feat = normalize(feat)
    label = knn.predict(feat)
    print("label: ")
    print(label)
    img_show = np.copy(img_gt)
    hit = 0
    cnt = 0
    tp = 0
    fp = 0
    fn = 0
    cnt = 0
    print("entrou no ultimo")
    for y in range(0, height, WINDOW_SIZE):
        for x in range(0, width, WINDOW_SIZE):
            if(label[cnt] == 1):
                img_show[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] = 0
            else:
                img_show[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] = 255
            for h in range(y, y+WINDOW_SIZE):
                for w in range(x, x+WINDOW_SIZE):
                    if(img_gt[h, w] == 255):
                        if(label[cnt] == 0): # É prédio e o knn previu como prédio
                            hit += 1
                            tp += 1
                        else: # Incorretamente previu que não é um prédio
                            fn += 1
                    else:
                        if(label[cnt] == 0): # Incorretamenpte previu que é um prédio
                            fp += 1
                        else: # Não é prédio e previu que não é prédio mesmo
                            hit += 1
            cnt += 1

    b = datetime.datetime.now()    
    print("Accuracy: %f"%(hit/(height*width)))
    print("IoU: %f"%(tp/(tp+fn+fp)))
    img_show = img_show.astype(np.uint8)
    cv.imwrite("prev_teste.tif", img_show)
    print("\nThe program took %d minutes and %d seconds to finish the classification"%(abs(b.minute-a.minute), abs(b.second-a.second)))

if __name__ == '__main__':
    main()
