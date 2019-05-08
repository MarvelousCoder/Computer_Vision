from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
import numpy as np
import os
import pickle

DIRNAME = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIRNAME, "../data/lfw-deepfunneled")


def feature_extractor():
    return Xception(weights="imagenet", include_top=False,
                    input_shape=(250, 250, 3), pooling='avg')


def extract_feature(model, image_path):
    img = image.load_img(image_path, target_size=(250, 250))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)

    return features


def load_people_dict():
    print("Loading Dictionary... This might take a while...")
    with open(os.path.join(DATA_PATH, "../people_dict.pkl"), "rb") as file:
        return pickle.load(file)


def build_people_dictionary(model):
    people = {}
    for person in os.listdir(DATA_PATH):
        people[person] = []
        for img in sorted(os.listdir(os.path.join(DATA_PATH, person))):
            print("Processing {}".format(img))
            people[person].append(extract_feature(model, os.path.join(DATA_PATH, person, img)))
    with open(os.path.join(DATA_PATH, "../people_dict.pkl"), "wb") as file:
        pickle.dump(people, file)


def load_data(filename, dictionary):
    X = []
    y = []
    with open(filename, 'r') as file:
        number = int(next(file))
        for _ in range(number):
            line = next(file)
            person, img_1, img_2 = line.split("\t")
            X.append((dictionary[person][int(img_1) - 1][0], dictionary[person][int(img_2) - 1][0]))
            y.append(1)
        for _ in range(number):
            line = next(file)
            person_1, img_1, person_2, img_2 = line.split("\t")
            X.append((dictionary[person_1][int(img_1) - 1][0], dictionary[person_2][int(img_2) - 1][0]))
            y.append(0)
    return np.array(X), np.array(y)
