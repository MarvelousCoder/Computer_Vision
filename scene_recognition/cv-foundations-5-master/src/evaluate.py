from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sn
import pandas as pd


DIRNAME = os.path.dirname(__file__)
TEST_DIR = os.path.join(DIRNAME, "../data/test/")


def evaluate(model, batch_size=2, confusionMatrix=False):
    datagen = ImageDataGenerator()
    test_generator = datagen.flow_from_directory(TEST_DIR,
                                                 (300, 250),
                                                 batch_size=batch_size,
                                                 shuffle=False
                                                 )
    Y_pred = model.predict_generator(test_generator, steps=test_generator.samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    acc = accuracy_score(test_generator.classes, y_pred)
    print("Acurácia: ", acc)

    if confusionMatrix:
        target_names = ["Bedroom", "Coast", "Forest", "Highway", "Industrial",
                        "InsideCity", "Kitchen", "LivingRoom", "Mountain",
                        "Office", "OpenCountry", "Store", "Street", "Suburb",
                        "TallBuilding"]
        cm = confusion_matrix(test_generator.classes, y_pred)
        cm = cm / np.sum(cm.astype(np.float), axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(np.round(cm, 2), index = [i for i in target_names],
                  columns = [i for i in target_names])
        print(classification_report(test_generator.classes, y_pred,
                                    target_names=target_names))
        fig = plt.figure(figsize=(10, 10))
        sn.heatmap(df_cm, annot=True)

def evaluate_bov(y_pred,y_label,name,confusionMatrix=False):
    acc = accuracy_score(y_label, y_pred)
    print("Acurácia: ", acc)

    if confusionMatrix:
        target_names = ["Bedroom", "Coast", "Forest", "Highway", "Industrial",
                        "InsideCity", "Kitchen", "LivingRoom", "Mountain",
                        "Office", "OpenCountry", "Store", "Street", "Suburb",
                        "TallBuilding"]
        cm = confusion_matrix(y_label, y_pred)
        cm = cm / np.sum(cm.astype(np.float), axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(np.round(cm, 2), index = [i for i in target_names],
                  columns = [i for i in target_names])
        fig = plt.figure(figsize=(10, 10))
        sn.heatmap(df_cm, annot=True)

        plt.savefig("{}_{}.png".format(name,acc))
