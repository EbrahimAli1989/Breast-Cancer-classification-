from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Concatenate, Flatten
from keras.models import Model
from keras import Input
from configuration.Configuration import pathes, Intilization as Config


class Multiscale:
    @staticmethod
    def build():
        # model 1
        input1 = Input(shape=(Config.mutiscalewidth1,Config.mutiscaleheight1,Config.inChannel))
        x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(input1)
        x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=3)(x)
        x = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=2)(x)

        # model 2
        input2 = Input(shape=(Config.mutiscalewidth2,Config.mutiscaleheight2,Config.inChannel))
        y = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(input2)
        y = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(y)
        y = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(y)
        y = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(y)
        y = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(y)
        y = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(y)
        y = MaxPooling2D(pool_size=2)(y)

        # model 3
        input3 = Input(shape=(Config.mutiscalewidth3,Config.mutiscaleheight3,Config.inChannel))
        z = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(input3)
        z = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(z)
        z = MaxPooling2D(pool_size=2)(z)
        z = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(z)
        z = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(z)
        z = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(z)
        z = MaxPooling2D(pool_size=2)(z)
        z = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(z)
        z = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(z)
        z = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(z)
        #y = MaxPooling2D(pool_size=2)(y)

        xyz = Concatenate(axis=3)([x, y, z])
        #xyz = Dropout(rate=0.5)(xyz)
        xyz = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(xyz)
        #xyz = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(xyz)
        xyz = MaxPooling2D(pool_size=2)(xyz)
        xyz = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(xyz)
        #xyz = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(xyz)
        xyz = MaxPooling2D(pool_size=2)(xyz)
        xyz = Conv2D(filters=1024, kernel_size=3, activation='relu', padding='same')(xyz)
        #xyz = Conv2D(filters=1024, kernel_size=3, activation='relu', padding='same')(xyz)
        xyz = Flatten()(xyz)
        xyz = Dropout(rate=0.1)(xyz)
        xyz = Dense(units=1024, activation='relu', name='r2')(xyz)
        xyz = Dense(units=Config.num_classes, activation='softmax', name='r3')(xyz)


        out = Model(inputs=[input1, input2, input3], outputs=xyz, name='r1')
        return out


if __name__ == '__main__':
    model = Multiscale()
    model.build()
    model.summary()
