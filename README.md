# Classification of Ultrasound Breast Images Using Fused Ensemble of Deep Learning Classifiers

Please cite this work if you use any codes in your work:

E. A. Nehary, S. Rajan, "Classification of Ultrasound Breast Images Using
Fused Ensemble of Deep Learning Classifiers," 2022 IEEE International Symposium on Medical Measurements and Applications (MeMeA), 2022, Accept

we propose a fusion of three models namely transfer learning, multi-scale and autoencoder. Transfer learning based on VGG16 model propose  to address limited dataset.
Autoencoder use to extract features from noisy images.  We propose a novel multiscale deep learning model to address learning of US images
with tumors of various sizes and shapes. Then, these three models fused using differential evolution (DE) algorithm to get the final results. 

## Prerequisites
### Download  Datasets :
The two datasets utilized in this work is available at:

https://academictorrents.com/details/d0b7b7ae40610bbeaea385aeb51658f527c86a16 and https://data.mendeley.com/datasets/wmy84gzngw/1

### Install the following packages:
* numpy
* scipy
* keras
* pillow
* tensorflow
* scikit-learn
* matplotlib
* scikit-plot

## Configuration:
[Configuration.py](Configuration.py) is the main file used for project configuration and setup. Please read it carefully and update the paths and other setup parameters.

## Training and Testing: 
To train and test each model indepndently run the following files:
* Transfer learning model:

```python
$ python3 Train_VGG16.py
```
* Multi-scale model:
``` python 
$ python3 Train_multiscale.py
```
* Autoencoder model:
``` python 
$ python3 Train_autoencoder.py
```

## Evaluation:
To evalute models using the ensemble combining strategy, you should run 
``` python 
$ python3 Ensamble_combining.py 
```
## License:
This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/EbrahimAli1989/Breast-Cancer-classification-/blob/main/LICENSE) file for details
