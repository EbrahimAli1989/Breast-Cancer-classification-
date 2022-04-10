from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape
from tensorflow.keras.layers import BatchNormalization, Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from tensorflow.keras.models import Model,Sequential
from configuration.Configuration import pathes, Intilization as Config

class autoencoder:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
    def encoder(self, input):
        # encoder
        #
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        encoded = BatchNormalization()(conv4)
        return encoded

    def decoder(self,encoder_output):
        # decoder
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        up1 = UpSampling2D((2, 2))(conv6)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        up2 = UpSampling2D((2, 2))(conv7)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)
        return decoded

    def fc(self,encoded):
        flat = Flatten()(encoded)
        den = Dense(1024, activation='relu')(flat) # first FC layer
        den = Dense(512, activation='relu')(den) # second FC layer
        out = Dense(self.num_classes, activation='softmax')(den) # Output layer
        return out
    def create_autoencoder(self):
        input_img = Input(shape=(Config.width, Config.height, Config.inChannel))
        return Model(input_img, self.decoder(self.encoder(input_img)))
    def Create_encoder_FC(self, autoencoderwieght):
        input_img = Input(shape=(Config.width, Config.height, Config.inChannel))
        encoded = self.encoder(input_img)
        full_model = Model(input_img, self.fc(encoded))
        autoencoder = Model(input_img, self.decoder(self.encoder(input_img)))
        autoencoder.load_weights(autoencoderwieght)
        for l1, l2 in zip(full_model.layers[:19], autoencoder.layers[0:19]):
            l1.set_weights(l2.get_weights())
        for layer in full_model.layers[0:19]:
            layer.trainable = False
        return full_model, autoencoder
    def Create_fullautoencoder(self):
        input_img = Input(shape=(Config.width, Config.height, Config.inChannel))
        encoded = self.encoder(input_img)
        full_model = Model(input_img, self.fc(encoded))
        return full_model
if __name__=='__main__':
    mdl = autoencoder()
    #model = mdl.create_autoencoder()
    model, _= mdl.Create_encoder_FC(autoencoderwieght=Config.save_Autoencoder_weight)
    model.summary()