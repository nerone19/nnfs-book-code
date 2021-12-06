
from nn_layers import *
from nn_loss import *
from accuracy import *
import sys
class Model:

    def __init__(self):
        self.layers = []
        self.softmax_clf_output = None

    def add(self,layer):
        self.layers.append(layer)
    
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss 
        self.optimizer = optimizer
        self.accuracy = accuracy

    def forward(self,X,training):
        self.input_layer.forward(X,training)

        for layer in self.layers:
            layer.forward(layer.prev.output,training)

        return layer.output

    def backward(self,output, y):
        
        if self.softmax_clf_output is not None:
            self.softmax_clf_output.backward(output,y)
            
            #the last output we got is from the softmax crossentropy loss 
            self.layers[-1].dinputs = self.softmax_clf_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
                
            return
        
            
            
        self.loss.backward(output,y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self,X,y, *, epochs = 1,print_every = 1,validation_data = None):

        self.accuracy.init(y)
        for epoch in range(1,epochs+1):

            output = self.forward(X, training=True)
            data_loss, regularization_loss = self.loss.calculate(output, y,use_regularization=True)
            loss = data_loss + regularization_loss
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output,y)  

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' + \
                f'acc: {accuracy:.3f}, ' +  \
                f'loss: {loss:.3f} (' + \
                f'data_loss: {data_loss:.3f}, ' + \
                f'reg_loss: {regularization_loss:.3f}), ' + \
                f'lr: {self.optimizer.current_learning_rate}')
                sys.stdout.flush()

        if validation_data:
            output = self.forward(X,training= False)
            data_loss = self.loss.calculate(output, y)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Print a summary
            print(f'validation, ' + \
            f'acc: {accuracy:.3f}, ' + \
            f'loss: {loss:.3f}')


    def finalize(self):

        self.input_layer = Layer_Input()
        self.trainable_layers = []
        Layer_count = len(self.layers)
        
        for i in range(Layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < Layer_count -1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
                
            if isinstance(self.layers[-1], Activation_Softmax) and \
                isinstance(self.loss, Loss_CategoricalCrossentropy):
                self.softmax_clf_output = Activation_Softmax_Loss_CategoricalCrossentropy() 

        self.loss.remember_trainable_layers(self.trainable_layers)



