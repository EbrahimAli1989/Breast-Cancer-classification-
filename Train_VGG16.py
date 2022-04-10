from tensorflow.keras.optimizers import Adadelta, RMSprop,SGD,Adam
from dataset.Loaddata import loaddata
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from configuration.Configuration import pathes, Intilization as Config
from models.Net_VGG16 import model_vgg16
from sklearn.metrics import classification_report, confusion_matrix



###### 1 - Loading the dataset #########
# retrieve the pathes to dataset
p = pathes()
p = p.returnpath()
# load dataset
l = loaddata(p)
x, y = l.load(target_size=(Config.width,Config.height))
# divide data to train and test
X, testX, Y, testY = train_test_split(x,y,test_size=0.2,shuffle=True, random_state=42, stratify=y)
trainX, valX, trainY, valY = train_test_split(X,Y,test_size=0.1,shuffle=True, random_state=42, stratify=Y)

print('Loading data Done*****************')

###### 2 - Loading the autoencoder model  #########
mdl = model_vgg16()
model = mdl.build()

print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
##### 3- Data Augmentation #######
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
##### 4- Start traning VGG16 ###########
print("[INFO] training head...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=Config.batch_size),
                    validation_data=(valX, valY), epochs=Config.epochs,
                    steps_per_epoch=len(trainX) // Config.batch_size, verbose=1)
# Save weight
model.save(Config.save_vgg16_weight)


##### 4- evaluate the  trained  VGG16 model  ###########
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1)))
print(confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1)))

