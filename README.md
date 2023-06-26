# Basic Image Classification with TensorFlow and Keras

The basics of Keras is used with TensorFlow as its backend to solve a basic image classification problem.
A neural network model is created, trained, and evaluated that after the training, predicts digits from hand-written images with a high degree of accuracy.

## Code and Resources Used
Python Version: 3.7

Packages: tensorflow, keras, matplotlib, numpy

## Problem to Solve
This graph describes the problem that we are trying to solve visually. We want to create and train a model that takes an image of a handwritten digit as input and predicts the class of that digit, that is, it predicts the digit or it predicts the class of the input image.


![image](https://github.com/AdeenQazi/image_classification/assets/72377270/806289cd-2396-446c-b0b2-5fbca6ba0187)

## DataSet
MNIST Dataset: The dataset consists of lots of images of handwritten digits along with their labels.

training data: 60,000 examples

test data: 10,000 examples

And each example has 28 rows and 28 columns (28 x 28 pixels)

## One Hot Encoding
After this encoding, every label will be converted to a list with 10 elements and the element at index to the corresponding class will be set to 1, rest will be set to 0:


![image](https://github.com/AdeenQazi/image_classification/assets/72377270/5bcaa72e-664d-4d7a-ad77-d3686f67c40a)

## Neural Networks

### Linear Equations

![image](https://github.com/AdeenQazi/image_classification/assets/72377270/3d448525-719e-43e9-b586-b3f87d1dbcad)

The above graph simply represents the equation:

y = w1 * x1 + w2 * x2 + w3 * x3 + b

Where the `w1, w2, w3` are called the weights and `b` is an intercept term called bias. The equation can also be *vectorised* like this:

y = W . X + b

Where `X = [x1, x2, x3]` and `W = [w1, w2, w3].T`. The .T means *transpose*. This is because we want the dot product to give us the result we want i.e. `w1 * x1 + w2 * x2 + w3 * x3`. This gives us the vectorised version of our linear equation.

A simple, linear approach to solving hand-written image classification problem - could it work?

![image](https://github.com/AdeenQazi/image_classification/assets/72377270/271a2060-72e0-48be-9283-abccc7887d3f)


### Neural Networks

![image](https://github.com/AdeenQazi/image_classification/assets/72377270/5511a58b-4c9b-4c52-825c-e0a4af514e3d)


This model is much more likely to solve the problem as it can learn more complex function mapping for the inputs and outputs in our dataset.

### Activation Functions

The first step in the node is the linear sum of the inputs:

Z = W . X + b

The second step in the node is the activation function output:

A = f(Z)

Graphical representation of a node where the two operations are performed:

![image](https://github.com/AdeenQazi/image_classification/assets/72377270/0bf1df97-5717-49d3-a2f1-2a3b65be9b1b)



## Creating Model
The model used is a Sequential class defined in keras and adding some layers to it. Two hidden layers are used with 128 nodes each, and one output layer with 10 nodes for the 10 classes (digits 0 - 9). All the layers will be dense layers (all the nodes of a layer will be connected with all the nodes of the preceding layer). 

Note: We do not need to start with input layer as for Sequential class the input example is the input layer, thus it is not defined explicitly.

We have used two activation functions;
1. relu -> acts a linear function for all the positive values, and sets all the negative values to 0
2. softmax -> gives probability scores for various nodes that sum up to 1, class with the highest probabilty gives the prediction.


## Model Evaluation

Training Data Accuracy: 95.98%

Test Data Accuracy: 96.06%

## Model Predictions

![image](https://github.com/AdeenQazi/image_classification/assets/72377270/c3e9217d-f59e-455e-b275-bde62286d837)


