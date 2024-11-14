# Classification Neural Network for Planets
This project was created as a submission for the UCSC AI club's SCAI competition. 

## Objective
This project creates two classification neural networks. Each of these neural networks takes in an image of a planet as input and classifies it as either Earth, Mars, Mercury, Neptune, Jupiter, Uranus, or Venus. The "Image Classification Planets" file creates a model with an accuracy of 98.85% and the "v2_planet_classifier_model" creates a model with an accuracy of 94%.  

## "Image Classification Planets" 
This file begins by importing the relevant dependencies. It then utilizes Tensorflow to process the data from the images of a data set downloaded from Kaggle. Then using sequential the file creates a neural network with 96 convolution filters with a kernal size of 3 by 3, a stride length of 1 by 1. It uses relu as the activation function. Lastly, the model is trained over 6 epochs and achieves a final accuracy of 98.85%.

## "v2_planet_classifier_model"
< greg insert a description here >
