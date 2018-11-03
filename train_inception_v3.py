# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from inception_v3 import Inception
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications import VGG16
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2 as cv
import os
import itertools

# nohup python3 train_inception_v3.py --dataset dataset/train --model output/inception.model --label-bin output/inception_lb.pickle --plot output/inception_plot.png > incept.out &

INIT_LR = 0.0001
EPOCHS = 50
BS = 32
WIDTH = 224
HEIGHT = 224

def plot_confusion_matrix(classes, y_test, y_pred):
	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)

	cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
	# Plot normalized confusion matrix
	plt.figure()
	plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f'
	thresh = cnf_matrix.max() / 2.
	for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
		plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
					color="white" if cnf_matrix[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.show()
	plt.savefig("output/confusion_matrix.png")

def split(image_path):
	data = []
	labels = []

	# loop over the input images
	for imagePath in image_path:
		# load the image, resize it (the required input
		# spatial dimensions of SmallVGGNet), and store the image in the
		# data list
		image = cv.imread(imagePath)
		image = cv.resize(image, (HEIGHT, WIDTH))
		data.append(image)

		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

	# scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)
	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	# (trainX, testX, trainY, testY) = train_test_split(data,
	# 	labels, test_size=0.25, random_state=42)

	return data, labels


def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True,
		help="path to input dataset of images")
	ap.add_argument("-m", "--model", required=True,
		help="path to output trained model")
	ap.add_argument("-l", "--label-bin", required=True,
		help="path to output label binarizer")
	ap.add_argument("-p", "--plot", required=True,
		help="path to output accuracy/loss plot")
	args = vars(ap.parse_args())

	# initialize the data and labels
	print("[INFO] loading images...")

	# grab the image paths and randomly shuffle them
	imagePaths = sorted(list(paths.list_images(args["dataset"])))
	random.seed(42)
	random.shuffle(imagePaths)
	trainX, trainY = split(imagePaths)

	imagePaths = sorted(list(paths.list_images("dataset/test")))
	random.seed(43)
	random.shuffle(imagePaths)
	testX, testY = split(imagePaths)

	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainY)
	testY = lb.transform(testY)

	# convert the labels from integers to vectors (for 2-class, binary
	# classification you should use Keras' to_categorical function
	# instead as the scikit-learn's LabelBinarizer will not return a
	# vector)

	# construct the image generator for data augmentation
	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
		horizontal_flip=True, fill_mode="nearest")

	model = Inception.build(width=WIDTH, height=HEIGHT, depth=3, classes=len(lb.classes_))

	# initialize the model and optimizer (you'll want to use
	# binary_crossentropy for 2-class classification)
	print("[INFO] training network...")
	opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

	# train the network
	H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
		epochs=EPOCHS)

	# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=BS)
	print(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=lb.classes_))
	print("Accuracy: {}".format(accuracy_score(testY.argmax(axis=1), predictions.argmax(axis=1))))
	print(confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1)))
	plot_confusion_matrix(lb.classes_, testY.argmax(axis=1), predictions.argmax(axis=1))

	# plot the training loss and accuracy
	N = np.arange(0, EPOCHS)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.plot(N, H.history["acc"], label="train_acc")
	plt.plot(N, H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy (SmallVGGNet)")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig(args["plot"])

	# save the model and label binarizer to disk
	print("[INFO] serializing network and label binarizer...")
	model.save(args["model"])
	f = open(args["label_bin"], "wb")
	f.write(pickle.dumps(lb))
	f.close()

if __name__ == '__main__':
	main()