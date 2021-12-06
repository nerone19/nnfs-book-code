import numpy as np
import nnfs
from nnfs.datasets import spiral_data,sine_data
from nn_layers import *
from nn_loss import *
from optimizer import *
from module import Model
from accuracy import *
nnfs.init()


X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)
# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case

# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4,
 bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())
# Set loss, optimizer and accuracy objects
model.set(
 loss=Loss_CategoricalCrossentropy(),
 optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-5),
 accuracy=Accuracy_Categorical()
)



# Finalize the model
model.finalize()
# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)