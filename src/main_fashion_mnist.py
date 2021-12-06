# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:30:18 2021

@author: gabri
"""
import os 
import urllib
import urllib.request
from zipfile import ZipFile
import os
import urllib
import urllib.request
import numpy as np
import cv2
import os

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
    print("Downloading {} and saving as {}...".format(URL,FILE))
    urllib.request.urlretrieve(URL, FILE)
    
print('Unzipping images...')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)
    
    

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
 # Scan all the directories and create a list of labels
 labels = os.listdir(os.path.join(path, dataset))
 # Create lists for samples and labels
 X = []
 y = []
 # For each label folder
 for label in labels:
 # And for each image in given folder
 for file in os.listdir(os.path.join(path, dataset, label)):
 # Read the image
 image = cv2.imread(os.path.join(path, dataset, label, file), 
cv2.IMREAD_UNCHANGED)
 # And append it and a label to the lists
 X.append(image)
 y.append(label)
 # Convert the data to proper numpy arrays and return
 return np.array(X), np.array(y).astype('uint8')
# MNIST dataset (train + test)
def create_data_mnist(path):
 # Load both sets separately
 X, y = load_mnist_dataset('train', path)
 X_test, y_test = load_mnist_dataset('test', path)
 # And return all the data
 return X, y, X_test, y_test
Thanks to this function, we can load in our data by doing:
# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')