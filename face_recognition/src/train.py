from utils import *
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Concatenate, Add, Subtract, Dot, Multiply
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import os


DIRNAME = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIRNAME, "../data/lfw-deepfunneled")

people_dict = load_people_dict()
X_train, y_train = load_data(os.path.join(DATA_PATH, "../pairsDevTrain.txt"), people_dict)
X_val, y_val = load_data(os.path.join(DATA_PATH, "../pairsDevTest.txt"), people_dict)

for name, merge in zip(["Cat", "Add", "Sub", "Dot", "mult"],
                       [Concatenate(), Add(), Subtract(), Dot(-1, True), Multiply()]):
    input_a = Input(shape=(2048,))
    input_b = Input(shape=(2048,))

    diff = merge([input_a, input_b])
    x = Dense(512, activation="relu")(diff)
    # x = Dropout(0.5)(x)
    # x = Dense(512, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)

    pred = Dense(1, activation="sigmoid")(x)

    model = Model([input_a, input_b], pred)

    # train
    rms = RMSprop()
    model.compile(loss="binary_crossentropy", optimizer=rms, metrics=["accuracy"])
    saver = ModelCheckpoint(os.path.join(DIRNAME, "../models/baseline") + "/model.hdf5", 
                            verbose=0,
                            save_best_only=True, monitor="val_acc",
                            mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5,
                                  patience=5, verbose=0, min_lr=0.0001)
    history = model.fit([X_train[:, 0], X_train[:, 1]], y_train,
                        batch_size=128,
                        epochs=60,
                        validation_data=([X_val[:, 0], X_val[:, 1]], y_val),
                        callbacks=[saver, reduce_lr],
                        verbose=0)

    model = load_model(os.path.join(DIRNAME, "../models/baseline/model.hdf5"))
    print("Resultados modelo {}".format(name))
    print("Validation loss e validation accuracy:")
    print(model.evaluate([X_val[:, 0], X_val[:, 1]], y_val))
    print("\n")

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'], label="Train" + name)
    plt.plot(history.history['val_acc'], label="Test" + name)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
plt.legend(loc="lower right", fontsize='x-small')
plt.show()
