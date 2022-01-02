from Utils.intial_information import pathes
from dataset.Loaddata import loaddata
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import imagenet_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from NetWorking.SqueezeNet import squeezenet
from tensorflow.keras.applications import VGG16
from keras.layers import Input
from tensorflow.keras.models import Model
from NetWorking.Multiscale_model import Multiscale, fusion_Multiscale
#from imutils import paths
from NetWorking.NetVGG import FCHeadNet
import numpy as np
import os
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
def generator_three_img(X1, X2, X3, y, batch_size):
    genX1 = aug.flow(X1, y,  batch_size=batch_size, seed=1)
    genX2 = aug.flow(X2, y, batch_size=batch_size, seed=1)
    genX3 = aug.flow(X3, y, batch_size=batch_size, seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        yield [X1i[0], X2i[0], X3i[0]], X1i[1]


def generator_three_feature(X1, X2, X3, y, batch_size):
    genX1 = aug.flow(X1, y, batch_size=batch_size, seed=1)
    genX2 = aug.flow(X2, y, batch_size=batch_size, seed=1)
    genX3 = aug.flow(X3, y, batch_size=batch_size, seed=1)
    dimension = 128  # 64, 96, 128
    model1 = squeezenet()
    model1 = model1.build(input_shape=(dimension, dimension, 3))

    model1.load_weights('squeezenet_model128.h5')
    model1.trainable = False
    last_layer1 = Model(inputs=model1.input, outputs=model1.output)
    last_layerfeature1 = Model(inputs=model1.input, outputs=model1.get_layer('pool5').output)

    dimension = 96  # 64, 96, 128
    model2 = squeezenet()
    model2 = model2.build(input_shape=(dimension, dimension, 3))

    model2.load_weights('squeezenet_model96.h5')
    model2.trainable = False
    last_layer2 = Model(inputs=model2.input, outputs=model2.output)
    last_layerfeature2 = Model(inputs=model2.input, outputs=model2.get_layer('pool5').output)

    dimension = 64  # 64, 96, 128
    model3 = squeezenet()
    model3 = model3.build(input_shape=(dimension, dimension, 3))

    model3.load_weights('squeezenet_model64.h5')
    model3.trainable = False
    last_layer3 = Model(inputs=model3.input, outputs=model3.output)
    last_layerfeature3 = Model(inputs=model3.input, outputs=model3.get_layer('pool5').output)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        out1 = last_layer1.predict(X1i[0])
        feature1 = last_layerfeature1.predict(X1i[0])

        out2 = last_layer2.predict(X2i[0])
        feature2 = last_layerfeature2.predict(X2i[0])

        out3 = last_layer3.predict(X3i[0])
        feature3 = last_layerfeature3.predict(X3i[0])

        yield [feature1, feature2, feature3], X1i[1]
def test_feature(X1, X2, X3, y):
    dimension = 128  # 64, 96, 128
    model1 = squeezenet()
    model1 = model1.build(input_shape=(dimension, dimension, 3))

    model1.load_weights('squeezenet_model128.h5')
    model1.trainable = False
    last_layer1 = Model(inputs=model1.input, outputs=model1.output)
    last_layerfeature1 = Model(inputs=model1.input, outputs=model1.get_layer('pool5').output)

    dimension = 96  # 64, 96, 128
    model2 = squeezenet()
    model2 = model2.build(input_shape=(dimension, dimension, 3))

    model2.load_weights('squeezenet_model96.h5')
    model2.trainable = False
    last_layer2 = Model(inputs=model2.input, outputs=model2.output)
    last_layerfeature2 = Model(inputs=model2.input, outputs=model2.get_layer('pool5').output)

    dimension = 64  # 64, 96, 128
    model3 = squeezenet()
    model3 = model3.build(input_shape=(dimension, dimension, 3))

    model3.load_weights('squeezenet_model64.h5')
    model3.trainable = False
    last_layer3 = Model(inputs=model3.input, outputs=model3.output)
    last_layerfeature3 = Model(inputs=model3.input, outputs=model3.get_layer('pool5').output)


    out1 = last_layer1.predict(X1)
    feature1 = last_layerfeature1.predict(X1)

    out2 = last_layer2.predict(X2)
    feature2 = last_layerfeature2.predict(X2)

    out3 = last_layer3.predict(X3)
    feature3 = last_layerfeature3.predict(X3)

    return [feature1, feature2, feature3], y
def predict_feature(X1, X2, X3):
    dimension = 128  # 64, 96, 128
    model1 = squeezenet()
    model1 = model1.build(input_shape=(dimension, dimension, 3))

    model1.load_weights('squeezenet_model128.h5')
    model1.trainable = False
    last_layer1 = Model(inputs=model1.input, outputs=model1.output)
    last_layerfeature1 = Model(inputs=model1.input, outputs=model1.get_layer('pool5').output)

    dimension = 96  # 64, 96, 128
    model2 = squeezenet()
    model2 = model2.build(input_shape=(dimension, dimension, 3))

    model2.load_weights('squeezenet_model96.h5')
    model2.trainable = False
    last_layer2 = Model(inputs=model2.input, outputs=model2.output)
    last_layerfeature2 = Model(inputs=model2.input, outputs=model2.get_layer('pool5').output)

    dimension = 64  # 64, 96, 128
    model3 = squeezenet()
    model3 = model3.build(input_shape=(dimension, dimension, 3))

    model3.load_weights('squeezenet_model64.h5')
    model3.trainable = False
    last_layer3 = Model(inputs=model3.input, outputs=model3.output)
    last_layerfeature3 = Model(inputs=model3.input, outputs=model3.get_layer('pool5').output)


    out1 = last_layer1.predict(X1)
    feature1 = last_layerfeature1.predict(X1)

    out2 = last_layer2.predict(X2)
    feature2 = last_layerfeature2.predict(X2)

    out3 = last_layer3.predict(X3)
    feature3 = last_layerfeature3.predict(X3)

    return [feature1, feature2, feature3]

#model = Multiscale()
model = fusion_Multiscale()
model = model.build()
model.summary()
p = pathes()
p = p.returnpath()
l = loaddata(p)
x1, y1 = l.load(target_size=(128,128))
x2, y2 = l.load(target_size=(96,96))
x3, y3 = l.load(target_size=(64,64))
print('Loading data Done*****************')

# augment datasets
# construct the image generator for data augmentation


trainX1, testX1, trainY1, testY1 = train_test_split(x1,y1,test_size=0.2,shuffle=True, random_state=42, stratify=y1)
trainX2, testX2, trainY2, testY2 = train_test_split(x2,y2,test_size=0.2,shuffle=True, random_state=42, stratify=y2)
trainX3, testX3, trainY3, testY3 = train_test_split(x3,y3,test_size=0.2,shuffle=True, random_state=42, stratify=y3)
# load the VGG16 network
#feature1, X2i, X3i, feature1= generator_three_feature(trainX1, trainX2, trainX3, trainY1, 32)
#xx= np.where(testY1 & testY2)
print("[INFO] loading network...")

print("[INFO] compiling model...")
opt = RMSprop(lr=0.0001)
model.compile(loss=["categorical_crossentropy","categorical_crossentropy","categorical_crossentropy"],
              optimizer=opt, metrics=["accuracy"])

# train the head of the network for a few epochs (all other
# layers are frozen) -- this will allow the new FC layers to
# start to become initialized with actual "learned" values
# versus pure random
print("[INFO] training head...")
it1 = aug.flow(trainX1, trainY1, batch_size=32)
ax1, ay1 = [], []
#for it in it1:
 #   images, labels = it
#    ax1.append(ax1)
 #   ay1.append(ay1)


it2 = aug.flow(trainX2, trainY2, batch_size=32)
it3 = aug.flow(trainX3, trainY3, batch_size=32)
it4 = aug.flow([trainX1, trainX2, trainX3], trainY1, batch_size=32)
model.fit_generator(generator_three_feature(trainX1, trainX2, trainX3, trainY1, 32),
                    epochs=100,
                    validation_data=test_feature(testX1,testX2,testX3,testY1),
                    steps_per_epoch=len(trainX1) // 32, verbose=1)

# evaluate the network after initialization
print("[INFO] evaluating after initialization...")
#model.save('multiscale_model.h5')
model.load_weights('multiscale_model.h5')
predictions = model.predict(predict_feature(testX1,testX2,testX3), batch_size=32)
np.savetxt('multiscale_predicationsqeeuze.csv', predictions, delimiter=',')
print(classification_report(testY1.argmax(axis=1),predictions.argmax(axis=1)))

print(confusion_matrix(testY1.argmax(axis=1),predictions.argmax(axis=1)))



