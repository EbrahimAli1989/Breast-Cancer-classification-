import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
from sklearn.preprocessing import LabelBinarizer
class loaddata:
    def __init__(self, pathes):
        self.pathes = pathes
    def load(self, target_size = (224,224),preprocess=None, dimension1D=False, img_netprocess=True, Binraize=True):
        images = list()
        labels = list()
        for p1 in self.pathes:
            print(p1)
            # return all files name
            files_and_directories = os.listdir(p1)
            for imname in files_and_directories:
                # return the file name with the path
                imagepath = os.path.join(p1, imname)
                # read image
                img = load_img(imagepath, target_size=target_size)
                img = img_to_array(img)
                #
                if img_netprocess:
                    img = imagenet_utils.preprocess_input(img)
                else:
                    img = img.astype('float32')
                    # scale from [0,255] to [-1,1]
                    img = (img - 127.5) / 127.5

                if dimension1D:
                    # return gary image
                    img = img[:,:,1]
                    img = np.expand_dims(img, axis=-1)

                images.append(img/255.0)
                # Extract the label
                labels.append(os.path.split(p1)[1])
        if Binraize:
            # convert labels to binary
            le = LabelBinarizer()
            labels = le.fit_transform(labels)
        print('Done')
        return np.array(images), labels

    def plot_training(self, H, N, plotPath):
        # construct a plot that plots and saves the training history
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(plotPath)
