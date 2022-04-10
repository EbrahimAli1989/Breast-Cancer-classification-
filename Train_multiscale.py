# import the necessary packages
from dataset.Loaddata import loaddata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from models.Multiscale_model import Multiscale
from configuration.Configuration import pathes, Intilization as Config

def generator_three_img(X1, X2, X3, y, batch_size, aug):
    genX1 = aug.flow(X1, y,  batch_size=batch_size, seed=1)
    genX2 = aug.flow(X2, y, batch_size=batch_size, seed=1)
    genX3 = aug.flow(X3, y, batch_size=batch_size, seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        yield [X1i[0], X2i[0], X3i[0]], X1i[1]

###### 1 - Loading the dataset #########
# retrieve the pathes to dataset
p = pathes()
p = p.returnpath()
# load dataset
l = loaddata(p)
# load with size 96 x 96
x1, y1 = l.load(target_size=(Config.mutiscalewidth1,Config.mutiscaleheight1))
# load with size 64 x 64
x2, y2 = l.load(target_size=(Config.mutiscalewidth2,Config.mutiscaleheight2))
# load with size 32 x 32
x3, y3 = l.load(target_size=(Config.mutiscalewidth3,Config.mutiscaleheight3))

# divide data to train and test
# 96 x 96
X, testX1, Y, testY1 = train_test_split(x1,y1,test_size=0.2,shuffle=True, random_state=42, stratify=y1)
trainX1, valX1, trainY1, valY1 = train_test_split(X,Y,test_size=0.1,shuffle=True, random_state=42, stratify=Y)
# 64 x 64
X, testX2, Y, testY2 = train_test_split(x2,y2,test_size=0.2,shuffle=True, random_state=42, stratify=y2)
trainX2, valX2, trainY2, valY2 = train_test_split(X,Y,test_size=0.1,shuffle=True, random_state=42, stratify=Y)
# 32 x 32
X, testX3, Y, testY3 = train_test_split(x3,y3,test_size=0.2,shuffle=True, random_state=42, stratify=y3)
trainX3, valX3, trainY3, valY3 = train_test_split(X,Y,test_size=0.1,shuffle=True, random_state=42, stratify=Y)
print('Loading data Done*****************')

###### 2 - Loading the Multiscale model  #########
mdl = Multiscale()
model = mdl.build()
print("[INFO] compiling model...")
opt = RMSprop(lr=0.0001)
model.compile(loss=["categorical_crossentropy","categorical_crossentropy","categorical_crossentropy"],
              optimizer=opt, metrics=["accuracy"])
##### 3- Data Augmentation #######
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

##### 4- Start traning Multiscale model ###########
model.fit_generator(generator_three_img(trainX1, trainX2, trainX3, trainY1, Config.batch_size, aug),
                    epochs=Config.epochs,
                    validation_data=([valX1,valX2,valX3],valY1),
                    steps_per_epoch=len(trainX1) // Config.batch_size, verbose=1)
# save model weight
model.save(Config.save_multiscale_weight)

##### 5- evaluate the  trained  Multiscale model  ###########
print("[INFO] evaluating after initialization...")
predictions = model.predict([testX1, testX2, testX3], batch_size=Config.batch_size)
print(classification_report(testY1.argmax(axis=1),predictions.argmax(axis=1)))
print(confusion_matrix(testY1.argmax(axis=1),predictions.argmax(axis=1)))

