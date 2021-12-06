# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:55:01 2021

@author: Administrator
"""
import numpy as np


class Layer_Input:
    #we need this method such that each layer can have a prev and next linked layer  
    def forward(self, inputs,training):
        self.output = inputs



# Dense layer
class Layer_Dense:
# Layer initialization
    def __init__(self, n_inputs, n_neurons,weight_regularizer_l1 = 0 ,weight_regularizer_l2 = 0,bias_regularizer_l1 = 0,bias_regularizer_l2 = 0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 =  weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
       

    # Forward pass
    def forward(self, inputs,training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        
        
        
    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.dweights * self.weight_regularizer_l1
        
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.dbiases * self.bias_regularizer_l1
            
        if self.weight_regularizer_l2 > 0:
            self.dweights += self.dweights * 2 * self.weight_regularizer_l2
        
        if self.bias_regularizer_l2 > 0:
            self.dbiases += self.dbiases * 2 * self.bias_regularizer_l2
            
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        

# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs,training):
        self.inputs = inputs
        
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
    def predictions(self, outputs):
        return outputs
        
class Dropout:
    def __init__(self,rate):
        self.rate = 1 - rate 
    def forward(self,inputs,training):
        self.input = inputs
        if not training:
            self.output = inputs.copy()
            return
        
        self.binary_mask = np.random.binomial(1,self.rate,size=inputs.shape)/ (self.rate)
        self.output = (inputs * self.binary_mask)

    def backward(self,dvalues):
        
        self.dinputs = dvalues * self.binary_mask
       
class activation_Sigmoid:
     # Forward pass
    def forward(self, inputs,training):
         self.output = 1 / (1 + np.exp(-inputs))
     # Backward pass
    def backward(self, dvalues):
         # Derivative - calculates from output of the sigmoid function
         self.dinputs = dvalues * (1 - self.output) * self.output
    def predictions(self,outputs):
        return (outputs > 0.5) * 1        
# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs,training):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
        keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
        keepdims=True)
        self.output = probabilities
    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
        enumerate(zip(self.output, dvalues)):
        # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)
    
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
            
            
    def predictions(self,outputs):
        return np.argmax(outputs,axis=-1) 
            
# Linear activation
class Activation_Linear:
    # Forward pass
    def forward(self, inputs,training):
         # Just remember values
         self.inputs = inputs
         self.output = inputs
     # Backward pass
    def backward(self, dvalues):
         # derivative is 1, 1 * dvalues = dvalues - the chain rule
         self.dinputs = dvalues.copy()
    def predictions(self, outputs):
        return outputs