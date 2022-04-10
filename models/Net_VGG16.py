# import the necessary packages
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers  import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from configuration.Configuration import pathes, Intilization as Config
from tensorflow.keras.models import Model,Sequential

class FCHeadNet:
    @staticmethod
    def build(baseModel, num_classes, D):
        # initialize the head model that will be placed on top of
        # the base, then add a FC layer
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        # add a softmax layer
        headModel = Dense(num_classes, activation="softmax", name='d1')(headModel)
        # return the model
        return headModel
class model_vgg16:
    def __init__(self):
        pass
    def build(self):
        # load the VGG16 network, ensuring the head FC layer sets are left  off
        baseModel = VGG16(weights="imagenet", include_top=False,
                          input_tensor=Input(shape=(Config.vggwidth, Config.vggwidth, Config.inChannel)))

        headModel = FCHeadNet.build(baseModel, Config.num_classes, 512)
        # place the head FC model on top of the base model -- this will
        # become the actual model we will train
        model = Model(inputs=baseModel.input, outputs=headModel)

        # loop over all layers in the base model and freeze them so they
        # will *not* be updated during the training process
        for layer in baseModel.layers:
            layer.trainable = False

        return model

if __name__ == '__main__':
    mdl = model_vgg16()
    model = mdl.build()
    model.summary()
    print('Done')

