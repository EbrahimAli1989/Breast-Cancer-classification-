from tensorflow.keras.models import load_model
from dataset.Loaddata import loaddata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from configuration.Configuration import pathes, Intilization as Config
from models.Autoencoder import autoencoder
from models.Net_VGG16 import model_vgg16
from Utils.utils import DE_algorithm, ensemble_predictions_prob


###### 1 - Loading the dataset #########
# retrieve the pathes to dataset
p = pathes()
p = p.returnpath()
# load dataset
l = loaddata(p)
# load with size 224 x 224
x0, y0 = l.load(target_size=(Config.vggwidth,Config.vggheight))
# load with size 96 x 96
x1, y1 = l.load(target_size=(Config.mutiscalewidth1,Config.mutiscaleheight1))
# load with size 64 x 64
x2, y2 = l.load(target_size=(Config.mutiscalewidth2,Config.mutiscaleheight2))
# load with size 32 x 32
x3, y3 = l.load(target_size=(Config.mutiscalewidth3,Config.mutiscaleheight3))

# divide data to train and test
# 224 x 224
X, testX224, Y, testY224= train_test_split(x0,y0,test_size=0.2,shuffle=True, random_state=42, stratify=y0)
trainX224, valX224, trainY224, valY224 = train_test_split(X,Y,test_size=0.1,shuffle=True, random_state=42, stratify=Y)
# 96 x 96
X, testX96, Y, testY96 = train_test_split(x1,y1,test_size=0.2,shuffle=True, random_state=42, stratify=y1)
trainX96, valX96, trainY96, valY96 = train_test_split(X,Y,test_size=0.1,shuffle=True, random_state=42, stratify=Y)
# 64 x 64
X, testX64, Y, testY64 = train_test_split(x2,y2,test_size=0.2,shuffle=True, random_state=42, stratify=y2)
trainX64, valX64, trainY64, valY64 = train_test_split(X,Y,test_size=0.1,shuffle=True, random_state=42, stratify=Y)
# 32 x 32
X, testX32, Y, testY32 = train_test_split(x3,y3,test_size=0.2,shuffle=True, random_state=42, stratify=y3)
trainX32, valX32, trainY32, valY32 = train_test_split(X,Y,test_size=0.1,shuffle=True, random_state=42, stratify=Y)

###### 2 - Loading  models  #########
# a- autoencoder model
mdl = autoencoder()
autoencoder_model = mdl.Create_fullautoencoder()
print(autoencoder_model.summary())
autoencoder_model.load_weights(Config.save_AutoFC_weight)
# b- transfer model
mdl = model_vgg16()
transfer_model = mdl.build()
transfer_model.load_weights(Config.save_vgg16_weight)
# c- Multiscale model 
#mdl = Multiscale()
#multiscale_model = mdl.build()
#multiscale_model.load_weights(Config.save_multiscale_weight)
multiscale_model= load_model(Config.save_multiscale_weight)
print(multiscale_model.summary())

print('Done')

###### 3 - Evalute each  model  #########
# a- autoencoder model
auto_pred_train = autoencoder_model.predict(trainX96, batch_size=Config.batch_size)
auto_pred_test = autoencoder_model.predict(testX96, batch_size=Config.batch_size)
# b- transfer model
tranf_pred_train = transfer_model.predict(trainX224, batch_size=Config.batch_size)
tranf_pred_test = transfer_model.predict(testX224, batch_size=Config.batch_size)
#c-Multiscale model
multi_pred_train = multiscale_model.predict([trainX96, trainX64, trainX32], batch_size=Config.batch_size)
multi_pred_test = multiscale_model.predict([testX96, testX64, testX32], batch_size=Config.batch_size)

###### 3 - Compute weights  p1, p2, and p3 using DE algorithm  #########
overall_predication = [auto_pred_train, tranf_pred_train, multi_pred_train]
optimizse_weights, equal_weights = DE_algorithm(pred=overall_predication, true_value= trainY224.argmax(axis=-1), n_members=2)
#
test_predication = [auto_pred_test, tranf_pred_test, multi_pred_test]
wiehted_yhatequal = ensemble_predictions_prob(equal_weights, test_predication)
wiehted_yhat = ensemble_predictions_prob(optimizse_weights, test_predication)

print('Equal weight....')
print(classification_report(testY224.argmax(axis=1), wiehted_yhatequal.argmax(axis=1)))
print(confusion_matrix(testY224.argmax(axis=1), wiehted_yhatequal.argmax(axis=1)))
print('Optimize weight...')
print(classification_report(testY224.argmax(axis=1), wiehted_yhat.argmax(axis=1)))
print(confusion_matrix(testY224.argmax(axis=1), wiehted_yhat.argmax(axis=1)))

print('Done')