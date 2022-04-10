class Intilization:
    batch_size = 32
    epochs = 100
    inChannel = 3
    width, height = 96, 96
    vggwidth, vggheight = 224, 224
    mutiscalewidth1, mutiscaleheight1 = 96, 96
    mutiscalewidth2, mutiscaleheight2 = 64, 64
    mutiscalewidth3, mutiscaleheight3 = 32, 32
    num_classes = 3
    save_Autoencoder_weight= "C:/Users/nehar/OneDrive/Desktop/Carleton/Medical Image Processing SYSC5304/project/Paper_code/Pretrained_model/autoencoder.h5"
    save_AutoFC_weight= "C:/Users/nehar/OneDrive/Desktop/Carleton/Medical Image Processing SYSC5304/project/Paper_code/Pretrained_model/autoencoder_classification.h5"
    save_vgg16_weight = "C:/Users/nehar/OneDrive/Desktop/Carleton/Medical Image Processing SYSC5304/project/Paper_code/Pretrained_model/vgg16_model.h5"
    save_multiscale_weight = "C:/Users/nehar/OneDrive/Desktop/Carleton/Medical Image Processing SYSC5304/project/Paper_code/Pretrained_model/multiscale_model.h5"

class pathes:
    def __init__(self):
        self.bayh_benign = 'C:/Users/nehar/OneDrive/Desktop/Carleton/Breast cancer research/Bayh data/Extracted images/benign'
        self.bayh_malignant = 'C:/Users/nehar/OneDrive/Desktop/Carleton/Breast cancer research/Bayh data/Extracted images/malignant'
        self.bayh_normal = 'C:/Users/nehar/OneDrive/Desktop/Carleton/Breast cancer research/Bayh data/Extracted images/normal'
        self.normal_artificial = 'C:/Users/nehar/OneDrive/Desktop/Carleton/Medical Image Processing SYSC5304/project/Code/normal'
        self.Mendely_benign = 'C:/Users/nehar/OneDrive/Desktop/Carleton/Breast cancer research/Breast Mendely/originals/benign'
        self.Mendely_malignant = 'C:/Users/nehar/OneDrive/Desktop/Carleton/Breast cancer research/Breast Mendely/originals/malignant'



    def returnpath(self):
        Pathes = list()
        Pathes.append(self.bayh_benign)
        Pathes.append(self.bayh_malignant)
        Pathes.append(self.bayh_normal)
        Pathes.append(self.Mendely_benign)
        Pathes.append(self.Mendely_malignant)
        return Pathes
