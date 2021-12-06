# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:55:53 2021

@author: Administrator
"""
import numpy as np
from nn_layers import *
# Common loss class
class Loss:

    def remember_trainable_layers(self, trainable_layers, use_regularization=False):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y,use_regularization =False):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        
        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        if use_regularization is False: 
            return data_loss
        # Return loss
        return data_loss, self.regularization_loss()
    
        # Regularization loss calculation
    def regularization_loss(self):
        # 0 by default
        regularization_loss = 0


        #train only not frozen layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                np.sum(np.abs(layer.weights))
            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                np.sum(layer.weights * \
                layer.weights)
            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                np.sum(np.abs(layer.biases))
            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                np.sum(layer.biases * \
                layer.biases)
        return regularization_loss
            
            
    
# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
            range(samples),
            y_true
            ]            

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clipped * y_true,
            axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        

        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
        
class BinaryCrossEntropy(Loss):
    
    # Forward pass
     def forward(self, y_pred, y_true):
         # Clip data to prevent division by 0
         # Clip both sides to not drag mean towards any value
         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
         # Calculate sample-wise loss
         sample_losses = -(y_true * np.log(y_pred_clipped) +
         (1 - y_true) * np.log(1 - y_pred_clipped))
         sample_losses = np.mean(sample_losses, axis=-1)
         # Return losses
         return sample_losses
     # Backward pass
     def backward(self, dvalues, y_true):
         # Number of samples
         samples = len(dvalues)
         # Number of outputs in every sample
         # We'll use the first sample to count them
         outputs = len(dvalues[0])
         # Clip data to prevent division by 0
         # Clip both sides to not drag mean towards any value
         clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
         # Calculate gradient
         self.dinputs = -(y_true / clipped_dvalues -
         (1 - y_true) / (1 - clipped_dvalues)) / outputs
         # Normalize gradient
         self.dinputs = self.dinputs / samples
         
class Loss_MeanSquaredError(Loss): # L2 loss
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        # Return losses
        return sample_losses
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
        
# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss): # L1 loss
     # Forward pass
     def forward(self, y_pred, y_true):
         # Calculate loss
         sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
         # Return losses
         return sample_losses
     # Backward pass
     def backward(self, dvalues, y_true):
         # Number of samples
         samples = len(dvalues)
         # Number of outputs in every sample
         # We'll use the first sample to count them
         outputs = len(dvalues[0])
         self.dinputs = np.sign(y_true - dvalues) / outputs
         # Normalize gradient
         self.dinputs = self.dinputs / samples

