# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:33:04 2021

@author: gabri
"""


class Model:
    
    def __init__(self):
        self.layers = []
        
    def add(self,layer):
        self.layers.append(layer)