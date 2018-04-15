# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 21:45:58 2018

@author: Chase
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import Adam,Nadam
 
def my_model():
    
    m = Sequential()
    
    #Conv Layer
    m.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(75, 75, 3)))
    m.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    m.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    m.add(Dropout(0.25))

    #Conv Layer2
    m.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    m.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    m.add(Dropout(0.10))

    #Conv layer 3
    m.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    m.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    m.add(Dropout(0.05))
#
#    #Conv block 4
#    m.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
#    m.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    #Dense layers
    m.add(Flatten())
    m.add(Dense(512, activation='relu'))
    m.add(Dropout(0.3))
    m.add(Dense(256, activation='relu'))
    m.add(Dropout(0.1))
    m.add(Dense(1, activation='sigmoid'))
    
    ##Declare our optimizer and compile the model. 
    optimizer = Adam() 
    m.compile(optimizer, loss='binary_crossentropy',
              metrics=['accuracy'])
    return m 