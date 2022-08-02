# Chest X-Ray CNN

## The Neural Network
This convolutional neural networks predicts whether a patient is normal or if they have COVID-19, pneumonia, or tuberculosis based on an x-ray of the patient's chest. The model will predict a list of 4 elements (3 indices), where each value in the list represents the probability that the image represents a chest with COVID-19, pneumonia, tuberculosis, or none of those diseases (represented by the "normal" class). In other words, given an input image, the model will output a list [*probability x-ray is COVID-19*, *probability x-ray is normal*, *probability x-ray is pneumonia*, *probability x-ray is tuberculosis*]. The element with the highest probability is the mons. Since both models are multiclass classification algorithms that predict categorical values, each uses a categorical crossentropy loss function and has 4 output neurons (one for each class). They use a standard SGD optimizer with a learning rate of 0.001 and have dropout layers to prevent overfitting.

1. The first model, found in the **melanoma_classifier.py** file, is a CNN that uses Tensorflow's ImageDataGenerator to augment the data it receives. It contains an architecture consisting of:
    - 1 Input layer (with an input shape of (128, 128, 3))
    - 1 Conv2D layer (with 32 filters, a kernel size of (3, 3), strides of (5, 5), and a ReLU activation function)
    - 1 Max pooling 2D layer (with a pooling size of (2, 2) and strides of (2, 2))
    - 1 Conv2D layer (with 64 filters, a kernel size of (3, 3), strides of (5, 5), and a ReLU activation function)
    - 1 Flatten layer
    - 1 Hidden layer (with 7 neurons and a ReLU activation function
    - 1 Output neuron (with 1 neuron and a sigmoid activation function)

2. The second model, found in the **melanoma_classifier_vgg16.py** file, uses the pretrained VGG16 base provided by Keras (these layers are untrained in the model) and only uses a horizontal flip layer to augment the data. It has an architecture of:
    - 1 Horizontal random flip layer (for image preprocessing)
    - 1 VGG16 base model (with an input shape of (128, 128, 3))
    - 1 Flatten layer
    - 1 Dropout layer (with a dropout rate of 0.3)
    - 1 Hidden layer (with 256 neurons and a ReLU activation function
    - 1 Output layer (with 1 output neuron and a sigmoid activation function)

I found that the VGG16 base model tends to get a slightly higher accuracy but takes significantly longer to train. Note that when running the VGG16 base model file, you will need to input the paths of the benign and malignant images for both the training and testing datasets (four paths total) as a string — the location for where to put the paths is signified in the **melanoma_classifier_vgg16.py** file with the words: 
- " < PATH TO MALIGNANT TRAIN IMAGES > " 
- " < PATH TO BENIGN TRAIN IMAGES > " 
- " < PATH TO MALIGNANT TEST IMAGES > " 
- " < PATH TO BENIGN TEST IMAGES > " 

When running the **melanoma_classifier.py** file, you will need to input the paths of the training and testing datasets (two paths total) as a string. These paths need to be inputted where the file reads:
- " < PATH TO TRAIN SET IMAGES > " 
- " < PATH TO TEST SET IMAGES > " 

Feel free to further tune the hyperparameters or build upon either of the models!

## The Dataset
The dataset used here can be found at this link: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images. Credit for the dataset collection goes to **Johnny T**, **Ahmed saber Elsheikhm**, **Gerry**, and others on *Kaggle*. The dataset contains approximately 9735 training images and 1000 testing images. Note that the images from the original dataset are resized to 128 x 128 images so that they are more manageable for the model. They are considered RGB by the model since the VGG16 model only accepts images with three color channels. The dataset is not included in the repository because it is too large to stabley upload to Github, so just use the link above to find and download the dataset.

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 

