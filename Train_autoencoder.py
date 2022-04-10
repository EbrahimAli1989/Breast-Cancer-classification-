from tensorflow.keras.optimizers import Adadelta, RMSprop,SGD,Adam
from dataset.Loaddata import loaddata
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from configuration.Configuration import pathes, Intilization as Config
from models.Autoencoder import autoencoder
from tensorflow.keras.losses import categorical_crossentropy
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
mdl = autoencoder()
model = mdl.create_autoencoder()
model.compile(loss='mean_squared_error', optimizer = RMSprop())

##### 3- Data Augmentation #######
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
##### 4- Start traning Autoencoder ###########
print("[INFO] training head...")
autoencoder_train = model.fit_generator(aug.flow(trainX, trainX, batch_size=Config.batch_size),
                    validation_data=(valX, valX), epochs=Config.epochs,
                    steps_per_epoch=len(trainX) // Config.batch_size, verbose=1)
# save wights
model.save_weights(Config.save_Autoencoder_weight)

##### 5- Training Encoder-FC classification model ###########
CoderFC_model, _ = mdl.Create_encoder_FC(autoencoderwieght=Config.save_Autoencoder_weight)
CoderFC_model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])

# start training
classify_train = CoderFC_model.fit_generator(aug.flow(trainX, trainY, batch_size=Config.batch_size),
                    validation_data=(testX, testY), epochs=Config.epochs,
                    steps_per_epoch=len(trainX) // Config.batch_size, verbose=1)
CoderFC_model.save_weights(Config.save_AutoFC_weight)

##### 5- Fine tuning Encoder-FC classification model ###########
for layer in CoderFC_model.layers[0:19]:
    layer.trainable = True

CoderFC_model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])

# retrain again
classify_train = CoderFC_model.fit_generator(aug.flow(trainX, trainY, batch_size=Config.batch_size),
                    validation_data=(testX, testY), epochs=Config.epochs,
                    steps_per_epoch=len(trainX) // Config.batch_size, verbose=1)
CoderFC_model.save_weights(Config.save_AutoFC_weight)

##### 4- evaluate the  trained  VGG16 model  ###########
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=Config.batch_size)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1)))
print(confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1)))


