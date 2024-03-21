
In this project, I set up a neural network that estimates the attitude of a drone from the measurements
of its Inertial Measurement Unit (IMU). The dataset consists of IMU readings from a quadcopter, provided by 
University *Professor De Croon, G.C.H.E.* 

Essentially, a Multi Layer Perceptron (MLP) is set up for learning regression. It is first applied to an arbitrary 
Sine signal, to check that it is working correctly, and then it is applied to the IMU data, using as input the drone
accelerations, and output its pitch. 
Data processing, and filtering is performed using domain knowledge.
Hyperparameters for the neural network can be tweaked to improve its performance

I worked on this individually, as part of my Artificial Intelligence course in BSc. Aerospace Engineering
Project took place: Feb 2023