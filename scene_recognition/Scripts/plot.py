import matplotlib.pyplot as plt
import numpy as np
import itertools

classes = ["Bedroom", "Coast", "Forest", "Highway", "Industrial",
                        "InsideCity", "Kitchen", "LivingRoom", "Mountain",
                        "Office", "OpenCountry", "Store", "Street", "Suburb",
                        "TallBuilding"]

cnf_matrix = np.array(
[[107,   0,   0,   0,   0,   0,   0,   9,   0,   0,   0,   0,   0,   0,   0],
 [  0, 245,   1,   2,   0,   0,   0,   0,   1,   0,  11,   0,   0,   0,   0],
 [  0,   0, 214,   0,   0,   0,   0,   0,   0,   0,  14,   0,   0,   0,   0],
 [  0,   0,   0, 153,   0,   1,   0,   0,   0,   0,   2,   1,   3,   0,   0],
 [  0,   0,   1,   2, 194,   3,   1,   0,   0,   0,   1,   1,   4,   2,   2],
 [  0,   0,   0,   0,   5, 185,   0,   0,   0,   0,   0,   4,   4,   7,   3],
 [  2,   0,   0,   0,   0,   0, 103,   2,   0,   1,   0,   2,   0,   0,   0],
 [  5,   0,   0,   0,   0,   0,   9, 175,   0,   0,   0,   0,   0,   0,   0],
 [  0,   2,   1,   0,   0,   0,   0,   0, 253,   0,  18,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   5,   0, 110,   0,   0,   0,   0,   0],
 [  0,  20,   6,   2,   0,   0,   0,   0,  10,   0, 272,   0,   0,   0,   0],
 [  1,   0,   0,   0,   1,   1,   0,   1,   0,   1,   0, 210,   0,   0,   0],
 [  0,   0,   0,   2,   2,   1,   0,   0,   0,   0,   0,   0, 186,   0,   1],
 [  0,   0,   0,   0,   0,   2,   0,   0,   0,   0,   0,   0,   0, 139,   0],
 [  0,   0,   4,   0,   5,   8,   0,   0,   0,   0,   0,   0,   2,   0, 237]]
)

np.set_printoptions(precision=2)

cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
# Plot normalized confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
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
plt.savefig("resultados/confusion_matrix_xception30.png")