# Classification Neural Network for Planets
This project was created as a submission for the UCSC AI club's SCAI competition. 

## Objective
This project creates two classification neural networks (Convolutional type). Each of these neural networks takes in an image of a planet as input and classifies it as either Earth, Mars, Mercury, Neptune, Jupiter, Uranus, or Venus. The "Image Classification Planets" file creates a model with an accuracy of 98.85% and the "v2_planet_classifier_model" creates a model with an accuracy of 94%.
Additionally, for this project, we also create and train a spiking neural network or SNN which has the same planet classification abilities as our other two models, only with increased efficiency.  

## "Image Classification Planets" 
This file begins by importing the relevant dependencies. It then utilizes Tensorflow to process the data from the images of a data set downloaded from Kaggle. Then using sequential the file creates a neural network with 96 convolution filters with a kernal size of 3 by 3, a stride length of 1 by 1. It uses relu as the activation function. Lastly, the model is trained over 6 epochs and achieves a final accuracy of 98.85%.

## "v2_planet_classifier_model"
The v2_planet_classifier_model is a CNN that utilizes TensorFlow and Keras, and a 60-20-20 split of training, testing, and validation data. It uses convolutional layers, maxpooling, and dropout for regularization. Data is augmented to improve results, horizontal flipping, and other cosmetic manipulations are used to add variation. The model utilizes the "relu" activation for the Convelution2D layer, and uses "softmax" for the final Dense layer. The model utilizes 32 convolution filters for the first layer, and 64 for the second layer. The model achieves an accuracy of roughly 94%, however testing on external images yielded an accuracy of roughly 40.% We believe perhaps more cosmetic manipulations are needed, or more variance in the training data can help improve accuracy.

## "SNN model"
The <insert model file name here> file holds the code for creating and training a spiking neural network model. 
