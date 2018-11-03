# import the necessary packages
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications import VGG16

class VGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = VGG16(weights='imagenet', include_top=False, input_shape=(height, width, 3))
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(15, activation='softmax')(x)

        model = Model(inputs=model.input, outputs=predictions)

        # for layer in model.layers:
        #     layer.trainable = False

        # for layer in model.layers[:249]:
        #     layer.trainable = False

        # for layer in model.layers[249:]:
        #     layer.trainable = True    
        # # return the constructed network architecture
        return model