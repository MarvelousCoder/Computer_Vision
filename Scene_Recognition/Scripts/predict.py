# import the necessary packages
from keras.models import load_model
import argparse
import pickle
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True,
	help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=224,
	help="target spatial dimension width")
ap.add_argument("-e", "--height", type=int, default=224,
	help="target spatial dimension height")
args = vars(ap.parse_args())

# load the input image and resize it to the target spatial dimensions
image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (args["width"], args["height"]))
image = image/255.0

# otherwise, we must be working with a CNN -- don't flatten the
# image, simply add the batch dimension
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# make a prediction on the image
preds = model.predict(image)

# find the class label index with the largest corresponding
# probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]
print(lb.classes_)
print(preds)
print(label)


# draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
	(0, 0, 255), 2)

# show the output image
cv2.imwrite("Image.png", output)
cv2.namedWindow("Previsao", cv2.WINDOW_NORMAL)
cv2.imshow("Previsao", output)
cv2.waitKey(0)
