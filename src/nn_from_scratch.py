# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:59:12 2021

@author: gabri
"""


import numpy as np 
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()


class Dense:

    def __init__(self,n_inputs, n_neurons):
        
        self.w = 0.10*np.random.randn(n_inputs, n_neurons)
        self.b = np.zeros((1, n_neurons))

    def forward(self,input):
        self.output = np.dot(input,self.w) + self.b  
        self.input = input
    def bacward(self, output):
        self.w.g = np.dot(self.input.T,output.g)
        self.input.g = np.dot(output.g, self.w.T)
        self.b = np.sum(output.g, axis= 0, keepdims =True) 
        
class ReLU:
    def forward(self,input):
        self.input = input
        self.output =  np.maximum(0,input)
    def bacward(self, output):
        self.output.g = output.copy()
        self.output.g[self.input <= 0] = 0
         

class Softmax: 
    def forward(self,inputs):
        self.input = inputs
        #we can obtain the same results of doing np.exp(input) which can cause overflow problems
        exp_values = np.exp(inputs - np.max(inputs,axis=1, keepdims = True))
        probabilities = exp_values/ np.sum(exp_values,axis=1, keepdims = True)
        self.outputs = probabilities
    def bacward(self, output):
        
        self.input.g = np.empty_like(output)
        
        #for each batch sampple, we compose pairs of softmax outptut and relative loss's gradient
        for index, (single_output, gradients) in zip(self.outputs, output.g):
            
            #flat the gradient (why?)
            single_output = single_output.reshape(-1,1)
            #softmax gradient. for each input we get probabilities for each label
            jacobian_matrix = np.diagflat(single_output) - (single_output*single_output.T)
            #we multiply the loss value with the jacobian matrix previously computed
            self.input.g[index] = np.jacobian_matrix,gradients



class Loss: 
    def calculate(self,y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        sample_losses = self.forward(y_pred, y_true)
        return np.mean(sample_losses)

class CrossEntropy(Loss):
    def forward(self,y_pred, y_true):
        len_samples = y_pred.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7) #we avoid 0(which can lead to exploding gradients) and 1(which leads to biased neurons)

        if( len(y_true.shape) == 1):
           conf_scores = y_pred_clipped[range(y_pred.shape[0]),y_true] #we access the confidence for the class indicated by the y_true array

        if( len(y_true.shape) == 2):
           conf_scores = np.sum(y_pred_clipped*y_true)
        negative_log_likelihoods = -np.log(conf_scores) 
        return negative_log_likelihoods
    
    def backward(self, y_pred,y_true):

        labels = len(self.y_pred[0])  
        if(len(self.y_true) == 1):
            y_true = np.eye(labels)[y_true]
        
        self.output.g = -y_true/y_pred
        #we normalize the gradient because otherwise it will become bigger as we increase the batch size (# samples)
        #we want the gradient will be the same even though the input size can change
        self.output.g /= len(y_pred)
    
    
def relu_grad(input,output):
    (input> 0).float() * output.g

def dense_grad(input, output,w,b):
    w.g = (input.g*output.g).sum(0)
    input.g = w.T*output.g
    b = output.g.sum(0)



X,y = spiral_data(samples=100, classes=3)
dense1 = Dense(2,3)
a1 = ReLU()
a2 = Softmax()
dense1.forward(X)
a1.forward(dense1.output)
a2.forward(a1.output)


a1.backward()

my_loss = CrossEntropy()
loss = my_loss.calculate(a2.outputs,y)
print(loss)




