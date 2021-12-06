# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 11:02:06 2021

@author: Administrator
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data,sine_data
from nn_layers import *
from nn_loss import *
from optimizer import *
nnfs.init()



def crossEntropy():
        
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    
    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 128,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    
    dropout1 = Dropout(0.1) 
    # Create second Dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = Layer_Dense(128, 3,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    # Perform a forward pass of our training data through this layer
    
    
    
    
    
    optimizer = OptimizerAdam(learning_rate=0.05,decay=5e-5)
    
    for epoch in range(10001):
        
        dense1.forward(X)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)
        
        dropout1.forward(activation1.output)
        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(dropout1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        data_loss = loss_activation.forward(dense2.output, y)
        
        #we calculate an additional regularization loss we will add to the normal one for higher penalization
        regularization_loss = loss_activation.regularization_loss(dense1) + loss_activation.regularization_loss(dense2)
        
        loss = data_loss + regularization_loss
        
        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)
        
        if(epoch % 1000 == 0):
            print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}), ' +
                    f'lr: {optimizer.current_learning_rate}')
                            
        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)
    
        
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
    
        #-------------------------------------------------------------------------#
        #VALIDATION STEP
        # Create test dataset
        X_test, y_test = spiral_data(samples=100, classes=3)
        # Perform a forward pass of our testing data through this layer
        dense1.forward(X_test)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)
        
        dropout1.forward(activation1.output)
        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(dropout1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        loss = loss_activation.forward(dense2.output, y_test)
        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y_test.shape) == 2:
            y_test = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions==y_test)
        if(epoch % 1000 == 0):
            print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
    
    # Print gradients
    # print(dense1.dweights)
    # print(dense1.dbiases)
    # print(dense2.dweights)
    # print(dense2.dbiases)

def binaryCrossEntropy():
            
    # Create dataset
    X, y = spiral_data(samples=100, classes=2)
    y = y.reshape(-11,1)
    
    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 128,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    
    dropout1 = Dropout(0.1) 
    # Create second Dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = Layer_Dense(128, 1,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
    activation2 = activation_Sigmoid()
    # Create Softmax classifier's combined loss and activation
    loss_function = BinaryCrossEntropy()
    # Perform a forward pass of our training data through this layer
    
    
    
    
    
    optimizer = OptimizerAdam(decay=5e-7)
    
    for epoch in range(10001):
        
        dense1.forward(X)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)

        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        activation2.forward(dense2.output)
        data_loss = loss_function.calculate(activation2.output, y)
        
        #we calculate an additional regularization loss we will add to the normal one for higher penalization
        regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
        
        loss = data_loss + regularization_loss
        
        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = (activation2.output > 0.5) *1
        accuracy = np.mean(predictions==y)
        
        if(epoch % 1000 == 0):
            print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}), ' +
                    f'lr: {optimizer.current_learning_rate}')
                            
        # Backward pass
        loss_function.backward(activation2.output, y)
        activation2.backward(loss_function.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
    
        
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
    
        #-------------------------------------------------------------------------#
        #VALIDATION STEP
        # Create test dataset
        X_test, y_test = spiral_data(samples=100, classes=2)
        y_test = y_test.reshape(-1,1)
        # Perform a forward pass of our testing data through this layer
        dense1.forward(X_test)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)

        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        activation2.forward(dense2.output)
        loss = loss_function.calculate(activation2.output, y_test)
        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = (activation2.output > 0.5) *1

        accuracy = np.mean(predictions==y_test)
        if(epoch % 1000 == 0):
            print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


def regression():
    # Create dataset
    X, y = sine_data()
    # Create Dense layer with 1 input feature and 64 output values
    dense1 = Layer_Dense(1, 64)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Create second Dense layer with 64 input features (as we take output
    # of previous layer here) and 64 output values
    dense2 = Layer_Dense(64, 64)
    # Create ReLU activation (to be used with Dense layer):
    activation2 = Activation_ReLU()
    # Create third Dense layer with 64 input features (as we take output
    # of previous layer here) and 1 output value
    dense3 = Layer_Dense(64, 1)
    # Create Linear activation:
    activation3 = Activation_Linear()
    # Create loss function
    loss_function = MeanSquaredLoss()
    # Create optimizer
    optimizer = OptimizerAdam(learning_rate=0.005, decay=1e-3)
    
    # Accuracy precision for accuracy calculation
    # There are no really accuracy factor for regression problem,
    # but we can simulate/approximate it. We'll calculate it by checking
    # how many values have a difference to their ground truth equivalent
    # less than given precision
    # We'll calculate this precision as a fraction of standard deviation
    # of all the ground truth values
    accuracy_precision = np.std(y) / 250
    # Train in loop
    for epoch in range(10001):
     # Perform a forward pass of our training data through this layer
     dense1.forward(X)
     # Perform a forward pass through activation function
     # takes the output of first dense layer here
     activation1.forward(dense1.output)
     # Perform a forward pass through second Dense layer
     # takes outputs of activation function
     # of first layer as inputs
     dense2.forward(activation1.output)
     # Perform a forward pass through activation function
     # takes the output of second dense layer here
     activation2.forward(dense2.output)
     # Perform a forward pass through third Dense layer
     # takes outputs of activation function of second layer as inputs
     dense3.forward(activation2.output)
     # Perform a forward pass through activation function
     # takes the output of third dense layer here
     activation3.forward(dense3.output)
     # Calculate the data loss
     data_loss = loss_function.calculate(activation3.output, y)
     # Calculate regularization penalty
     regularization_loss = \
     loss_function.regularization_loss(dense1) + \
     loss_function.regularization_loss(dense2) + \
     loss_function.regularization_loss(dense3)
     # Calculate overall loss
     loss = data_loss + regularization_loss

# Calculate accuracy from output of activation2 and targets
     # To calculate it we're taking absolute difference between
     # predictions and ground truth values and compare if differences
     # are lower than given precision value
     predictions = activation3.output
     accuracy = np.mean(np.absolute(predictions - y) <
     accuracy_precision)
     if not epoch % 100:
                 print(f'epoch: {epoch}, ' +
                 f'acc: {accuracy:.3f}, ' +
                 f'loss: {loss:.3f} (' +
                 f'data_loss: {data_loss:.3f}, ' +
                 f'reg_loss: {regularization_loss:.3f}), ' +
                 f'lr: {optimizer.current_learning_rate}')
     # Backward pass
     loss_function.backward(activation3.output, y)
     activation3.backward(loss_function.dinputs)
     dense3.backward(activation3.dinputs)
     activation2.backward(dense3.dinputs)
     dense2.backward(activation2.dinputs)
     activation1.backward(dense2.dinputs)
     dense1.backward(activation1.dinputs)
     # Update weights and biases
     optimizer.pre_update_params()
     optimizer.update_params(dense1)
     optimizer.update_params(dense2)
     optimizer.update_params(dense3)
     optimizer.post_update_params()
    import matplotlib.pyplot as plt
    X_test, y_test = sine_data()
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    plt.plot(X_test, y_test)
    plt.plot(X_test, activation3.output)
    plt.show()

regression()






# binaryCrossEntropy()