# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2 as cv
import os
import itertools
from vgg16_places_365 import VGG16_Places365

# python3 train_vgg16_places.py --training dataset/train --test dataset/test --plot output/vgg16_places.png

INIT_LR = 0.001
EPOCHS = 50
BS = 32
WIDTH = 250
HEIGHT = 300

def plot_confusion_matrix(classes, y_test, y_pred):
	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)

	cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
	# Plot normalized confusion matrix
	plt.figure(figsize=(10, 10))
	plt.imshow(cnf_matrix, interpolation='nearest')
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.yticks(tick_marks, classes)

	fmt = '.2f'
	thresh = cnf_matrix.max() / 2.
	for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
		plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
					color="white" if cnf_matrix[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.savefig("output/confusion_matrix_vgg16.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True,
        help="path to input dataset of training images")
    ap.add_argument("-tt", "--test", required=True,
        help="path to input dataset of training images")
    ap.add_argument("-p", "--plot", required=True,
        help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    # initialize the data and labels
    print("[INFO] loading images...")

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.2,
                                 rotation_range=30,
                                 fill_mode="nearest",
                                 validation_split=0.2)

    train_generator = datagen.flow_from_directory(args["training"],
                                                  (HEIGHT, WIDTH),
                                                  batch_size=BS,
                                                  subset='training')

    valid_generator = datagen.flow_from_directory(args["training"],
                                                (HEIGHT, WIDTH),
                                                  batch_size=BS,
                                                  subset='validation')

    # initialize our VGG-like Convolutional Neural Network
    base_model = VGG16_Places365(include_top=False, weights='places',
                                     input_shape=(HEIGHT, WIDTH, 3),
                                     pooling="avg")
    x = base_model.output
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(15, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # initialize the model and optimizer (you'll want to use
    # binary_crossentropy for 2-class classification)
    print("[INFO] training network...")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS, clipnorm = 5., momentum = 0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    H = model.fit_generator(train_generator, validation_data=valid_generator,
                            steps_per_epoch=train_generator.samples // BS + 1, 
                            validation_steps=valid_generator.samples // BS + 1, 
                            epochs=EPOCHS)

    ##########################
    ## EVALUATE THE NETWORK ##
    ##########################
    print("[INFO] evaluating network...")
    datagen = ImageDataGenerator()
    test_generator = datagen.flow_from_directory(args["test"],
                                                (HEIGHT, WIDTH),
                                                 batch_size=BS,
                                                 shuffle=False)
    Y_pred = model.predict_generator(test_generator, steps=test_generator.samples // BS + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    acc = accuracy_score(test_generator.classes, y_pred)
    print("Accuracy: ", acc)

    # plot the training loss and accuracy
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy (VGG16Places365)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["plot"])

    target_names = ["Bedroom", "Coast", "Forest", "Highway", "Industrial",
                        "InsideCity", "Kitchen", "LivingRoom", "Mountain",
                        "Office", "OpenCountry", "Store", "Street", "Suburb",
                        "TallBuilding"]
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))
    plot_confusion_matrix(target_names, test_generator.classes, y_pred)

if __name__ == '__main__':
	main()