import matplotlib.pyplot as plt

import numpy as np
from imutils import paths
import cv2 as cv
import os
import sys
import matplotlib.pyplot as plt
import random
from local_binary_patterns import LocalBinaryPatterns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def main():
    # train_set = sorted(list(paths.list_images("data/images")))
    # random.shuffle(train_set)
    # _, Y_train = get_dataset(train_set)

    pairs = read_pairs(os.path.expanduser("data/pairsDevTrain.txt"))
    train_set, _ = get_paths(os.path.expanduser("data/images"), pairs)
    X_train, Y_train = get_dataset(train_set)
    print("treino foi")

    pairs = read_pairs(os.path.expanduser("data/pairsDevTest.txt"))
    validation_set, _ = get_paths(os.path.expanduser("data/images"), pairs)
    X_val, Y_val = get_dataset(validation_set)
    print("val tmb")

    model = LinearSVC(C=100.0, random_state=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_val)
    print(accuracy_score(Y_val, pred))

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def get_dataset(image_path):
    dataset = []
    labels = []
    desc = LocalBinaryPatterns(24, 8)
    for img in image_path:
        image = cv.imread(img)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        dataset.append(hist)
        label = img.split(os.path.sep)[-2]
        labels.append(label)

    dataset = np.array(dataset, dtype="float")
    dataset = normalize(dataset)
    labels = np.array(labels)
    return dataset, labels

if __name__ == "__main__":
    main()