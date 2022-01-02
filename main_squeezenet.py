from Utils.intial_information import pathes
from dataset.Loaddata import loaddata
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import imagenet_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
# import the necessary packages
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from NetWorking.Multiscale_model import Multiscale
from NetWorking.SqueezeNet import squeezenet
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
    genX1 = aug.flow(X1, y,  batch_size=batch_size, seed=1)
    genX2 = aug.flow(X2, y, batch_size=batch_size, seed=1)
    genX3 = aug.flow(X3, y, batch_size=batch_size, seed=1)
    dimension = 128  # 64, 96, 128
    model1 = squeezenet()
    model1 = model1.build(input_shape=(dimension, dimension, 3))

    model1.load_weights('squeezenet_model128.h5')
    model1.trainable = False
    last_layer1 = Model(inputs=model1.input, outputs=model1.output)
    last_layerfeature1 = Model(inputs=model1.input, outputs=model1.get_layer('pool5').output)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        out1 = last_layer1.predict(X1i)
        feature1 = last_layerfeature1.predict(X1i)

        yield [feature1[0], X2i[0], X3i[0]], feature1[1]
def generator_one_img(X1, y, batch_size):
    genX1 = aug.flow(X1, y,  batch_size=batch_size, seed=1)
    while True:
        X1i = genX1.next()
        yield X1i[0], X1i[1]

#model = Multiscale()
dimension = 128  #64, 96, 128
model = squeezenet()
model = model.build(input_shape=(dimension,dimension,3))
model.summary()
p = pathes()
#p = p.returnpath()
p = p.returnpathAll()
l = loaddata(p)
x1, y1 = l.load(target_size=(dimension,dimension))
#x2, y2 = l.load(target_size=(64,64))
#x3, y3 = l.load(target_size=(32,32))
print('Loading data Done*****************')

# augment datasets
# construct the image generator for data augmentation


trainX1, testX1, trainY1, testY1 = train_test_split(x1,y1,test_size=0.2,shuffle=True, random_state=42)
#trainX2, testX2, trainY2, testY2 = train_test_split(x2,y2,test_size=0.2,shuffle=True, random_state=42)
#trainX3, testX3, trainY3, testY3 = train_test_split(x3,y3,test_size=0.2,shuffle=True, random_state=42)
# load the VGG16 network
#xx= np.where(testY1 & testY2)
print("[INFO] loading network...")

print("[INFO] compiling model...")
opt = RMSprop(lr=0.0001)
model.compile(loss="categorical_crossentropy",
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


#it2 = aug.flow(trainX2, trainY2, batch_size=32)
#it3 = aug.flow(trainX3, trainY3, batch_size=32)
#it4 = aug.flow([trainX1, trainX2, trainX3], trainY1, batch_size=32)
model.fit_generator(generator_one_img(trainX1, trainY1, 32),
                    epochs=200,
                    validation_data=(testX1,testY1),
                    steps_per_epoch=len(trainX1) // 32, verbose=1)
# evaluate the network after initialization
print("[INFO] evaluating after initialization...")
model.save('squeezenet_modelartifical'+str(dimension)+'.h5')
predictions = model.predict(testX1, batch_size=32)
print(classification_report(testY1.argmax(axis=1),predictions.argmax(axis=1)))

print(confusion_matrix(testY1.argmax(axis=1),predictions.argmax(axis=1)))

'''

########## fusion phase
dimension = 128  #64, 96, 128
model = squeezenet()
model = model.build(input_shape=(dimension,dimension,3))

model.load_weights('squeezenet_model128.h5')
model.trainable=False
print("weights:", len(model.weights))
print("trainable_weights:", len(model.trainable_weights))
print("non_trainable_weights:", len(model.non_trainable_weights))

image_batch = np.ones((5,128,128,3),np.float32)
last_layer1 = Model(inputs=model.input, outputs=model.output)
last_layerfeature1 = Model(inputs=model.input, outputs=model.get_layer('pool5').output)
out1 = last_layer1.predict(image_batch)
feature1 = last_layerfeature1.predict(image_batch)
conv1 = Conv2D(256, 3, activation='relu')(last_layerfeature1)
conv2 =  Conv2D(256, 3, activation='relu')(conv1)

Mymodel = Model(inputs=last_layerfeature1, outputs = conv2)
#####################
dimension = 96  #64, 96, 128
model = squeezenet()
model = model.build(input_shape=(dimension,dimension,3))

model.load_weights('squeezenet_model96.h5')
model.trainable=False
print("weights:", len(model.weights))
print("trainable_weights:", len(model.trainable_weights))
print("non_trainable_weights:", len(model.non_trainable_weights))

image_batch = np.ones((5,96,96,3),np.float32)
last_layer2 = Model(inputs=model.input, outputs=model.output)
last_layerfeature2 = Model(inputs=model.input, outputs=model.get_layer('pool5').output)
out2 = last_layer2.predict(image_batch)
feature2 = last_layerfeature2.predict(image_batch)
################################
dimension = 64  #64, 96, 128
model = squeezenet()
model = model.build(input_shape=(dimension,dimension,3))

model.load_weights('squeezenet_model64.h5')
model.trainable=False
print("weights:", len(model.weights))
print("trainable_weights:", len(model.trainable_weights))
print("non_trainable_weights:", len(model.non_trainable_weights))

image_batch = np.ones((5,64,64,3),np.float32)
last_layer3 = Model(inputs=model.input, outputs=model.output)
last_layerfeature3 = Model(inputs=model.input, outputs=model.get_layer('pool5').output)
out3 = last_layer3.predict(image_batch)
feature3 = last_layerfeature3.predict(image_batch)
'''
print('Done...')
#out1 = last_layer1.predict(image_batch)