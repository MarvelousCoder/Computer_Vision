# USAGE
# python3 extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle \
#	--detector face_detection_model --embedding-model openface_nn4.small2.v1.t7
# python3 extract_embeddings.py --dataset turma --embeddings output/embeddings_turma.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# carrega nosso serialized face detector do disco
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# carrega nosso face embedding model do disco
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# pega os caminhos para imagens de entrada no dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# inicializa nossas listas de facial embeddings extraídas e as labels correspondentes
knownEmbeddings = []
knownNames = []

# inicializa o número total de faces processadas
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# carrega a imagem, dá um resize para ter uma largura de 600 pixels
	# (mantendo a aspect ratio), e então pega as dimensões da imagem.
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# constrói um blob da imagem
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# aplica o face detector baseado em deep learning da OpenCV para localizar
	# rostos na imagem de entrada 
	detector.setInput(imageBlob)
	detections = detector.forward()

	# garante que pelo menos uma face foi encontrada
	if len(detections) > 0:
		# Assume-se que cada imagem possui apenas um rosto, então encontra a bounding box
		# com a maior probabilidade
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# Garante que a detecção com a maior probabilidade também significa
		# nossa teste mínimo de probabilidade (filtra detecções fracas)
		if confidence > args["confidence"]:
			# computa as coordenadas da bounding box para o rosto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extrai a ROI da face e pega as dimensões
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# garante que a largura e altura do rosto são grandes o suficiente
			if fW < 20 or fH < 20:
				continue

			# constrói um blob para a ROI da face, então passa o
			# blob no face embedding model para obter a 128-d 
			# quantificação da face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# Adiciona o nome da pessoa + face embedding para as respectivas listas
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# salva as embeddings faciais + os nomes no disco
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()