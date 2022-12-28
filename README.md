# SML

[![Build Status](https://github.com/SammithSB/SML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SammithSB/SML.jl/actions/workflows/CI.yml?query=branch%3Amain)

SML is a machine learning package written Julia, built with intention to brush up on machine learning skills and learning Julia.

## 1 Linear Regression 

Linear regression is a statistical model to model a linear relatinship between dependant and independant variable. It is a supervised machine learning model that is used to predict continous variable outcome based on the input features. Can say, the main goal of this model is to prerdict the line of best fit for the input features. It is optimised by minimising the mean square error between predicted value and true value. This line of best fit is defined by the equation ```y=mx+b```, where ```w``` is the weight of coefficient, ```x``` is the input feature vector, ```b``` being the bias term and ```y``` being the predicted output.

The weight and bias of the equation is optimised usign gradient descent, it is an algorithm that's goal is to find minima of the function, it keeps moving in the direction that reduces the gradient until it finds the optimal values. It updates the weight and bias in the opposite direction of increase after taking the gradient, the size of update is measured by learning rate, which is a hyperparamter that decides how fast algorithm converges to minimum, if learning rate is very high, its very unlikely we will reach the minimum and at the same time if the learning rate is too low, it will take us a lot of iterations to reach optimal values.

We start with zeros for both weights and bias, we iterate for a certain number of epochs, in each epoch we calcualte ```y_pred``` using ```w``` and ```b``` and the gradient ```dw``` and ```db```, which are then again used to update `w` and `b`.

There is another issue that machine learning algorithms face when there is shortage of data which is overfitting i.e, it is is giving too much importance to input features and is giving amazing accuracy but when the same is used for new data, the accuracy is very bad and to prevent this from happening we used regularisation where add a penalty term which will encourage model to choose simpler models for predictions which can be generalised well. The penalty term is controlled by the hyperparameter `reg_lambda`. We use L2 regularisation in this code, which adds a penalty term to the loss function that is proportional to the sum of the squares of the weights. To implement L2 regularization, the code computes the regularization penalty term reg_loss as the sum of the squares of the weights multiplied by reg_lambda and divided by the number of samples. This is then added to mean squared loss which becomes the total loss.`reg_lambda` can be changed according to need how strong the penalty needs to be.

Finally, the predict function is used to generate predictions given a set of input features `X` and the optimized weight and bias parameters `w` and `b`. It computes the dot product between `X` and `w`, adds the bias term `b`, and returns the resulting vector of predictions.
