# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:56:52 2021

@author: Administrator
"""


#NOTE: 
#the square root take care of the negative numbers and avoid the number will grow too much and too fast



import nnfs
from nn_layers import *
nnfs.init()


# SGD optimizer
class Optimizer:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1e-4, decay=0., momentum=0.,epsilon=1e-7,rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.epsilon = epsilon
        self.rho = rho
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class OptimizerSGD(Optimizer):
        # Update parameters
    def update_params(self, layer):
        if(self.momentum):
            if not hasattr(layer,'weight_momentums'):
                layer.weight_momentums = np.zeros(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.bias)
                
            weight_updates = self.weight_momentums * layer.weight_momentums - \
                                layer.dweights * layer.learning_rate
            bias_updates = self.bias_momentums * layer.bias_momentums - \
                                layer.dbiases * layer.learning_rate
        else: 
            weight_updates += -self.current_learning_rate * layer.dweights
            bias_updates += -self.current_learning_rate * layer.dbiases
            
        layer.weights += weight_updates
        layer.biases += bias_updates

class OptimizerAdagrad(Optimizer):
    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases) 
            
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        
        layer.weights += - layer.dweights * self.learning_rate / np.sqrt(layer.weight_cache) + self.epsilon
        layer.biases += - layer.dbiases * self.learning_rate / np.sqrt(layer.bias_cache) + self.epsilon


class OptimizerRmsprop(Optimizer):
    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases) 
        
        #it is like a moving average for the gradient but focusing on caches. The latter moves together with data and learning does not stall
        #We always look behind to old caches but we also keep into account the current gradient
            #rho is the decay rate of our  weight cache
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) *  layer.dweights**2
        layer.bias_cache =  self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases**2
        
        layer.weights += - layer.dweights * self.current_learning_rate / np.sqrt(layer.weight_cache) + self.epsilon
        layer.biases += - layer.dbiases * self.current_learning_rate / np.sqrt(layer.bias_cache) + self.epsilon


class OptimizerAdam(Optimizer):
    def __init__(self, learning_rate=1e-4, decay=0.,epsilon=1e-7,beta_1=0.9,beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    # Call once before any parameter updates
    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases) 
    
        layer.weight_momentum = (self.beta_1) * layer.weight_momentum + (1 - self.beta_1)*layer.dweights
        layer.bias_momentum = (self.beta_1) * layer.bias_momentum + (1 - self.beta_1)*layer.dbiases
        
        weight_momentum_corrected = layer.weight_momentum /\
                                         ( 1- self.beta_1**(self.iterations+1) )
        
        bias_momentum_corrected = layer.bias_momentum /\
                                        ( 1- self.beta_1**(self.iterations+1) )
       
        #inspiration from RMSPROP
        layer.weight_cache = self.beta_2 * layer.weight_cache +  (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache +  (1 - self.beta_2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache /\
            (1 - self.beta_2 ** (self.iterations+1))
        bias_cache_corrected = layer.bias_cache /\
            (1 - self.beta_2 ** (self.iterations+1))

        #inspiration from SGD
        layer.weights += - self.current_learning_rate * weight_momentum_corrected / \
                                (np.sqrt(weight_cache_corrected) +self.epsilon)
        layer.biases += - self.current_learning_rate * bias_momentum_corrected / \
                                (np.sqrt(bias_cache_corrected) +self.epsilon)
