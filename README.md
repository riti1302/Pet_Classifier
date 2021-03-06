# Pet Classifier

The original cat-vs-dog dataset from [kaggle](https://www.kaggle.com/c/dogs-vs-cats) consist of 25000 training images. I have used only 6000 for training set, 3000 images for validation set and 3000 images for test set.

### Dependencies
* Google colab
* Keras
* Python 3.6
* Matplotlib
* Numpy

I have used 3 different Convolutional Neural Network models for this classifier:
1. [Pet_Classifier_without_augmentation](Pet_Classifier_without_augmentation.ipynb)- It consists of bunch of Convolution and Pooling layers, all trained from scratch. It gives an accuracy of **76.96%** on 30 epochs on test dataset.   
2. [Pet_Classifier_with_augmentation](Pet_Classifier_with_augmentation.ipynb)- It uses ImageDataGenerator along with a model similar   to that of the first model. It gives an accuracy of **83.63%** on 30 epochs on test dataset.  
3. [Pet_Classifier_using_pre-trained_model](Pet_Classifier_using-pre_-trained_model.ipynb)- It uses a pre-trained VGG16 model, initially with a self-trained classifier, and again with fine tuning of the fifth convolutional block and the self-trained classifier. This gives **94.39% accuracy** on 30 epochs.
