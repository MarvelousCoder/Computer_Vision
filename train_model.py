# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle
# python3 train_model.py --embeddings output/embeddings_turma.pickle --recognizer output/recognizer_turma.pickle --le output/le_turma.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# carrega as face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# treina o modelo usado para aceitar as 128-d embeddings da face e então
# produzir o reconhecimento de face
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# salva o modelo atual de reconhecimento de face
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# salva o label encoder
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()