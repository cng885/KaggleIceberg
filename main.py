# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 21:08:51 2018

@author: Chase
"""

import numpy as np
import pandas as pd

##Import My Classes
import util
import model 

##Keras Imports
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


##Constant's 
epochs = 10

df_train = pd.read_json('input/train.json')
df_train.inc_angle = df_train.inc_angle.replace('na',0)
df_train = df_train[df_train.inc_angle>0]

train_y = np.array(df_train['is_iceberg'])

train_scaled = util.get_scaled_imgs(df_train)

x_train, x_test, y_train, y_test = train_test_split(train_scaled, train_y,
                                                    test_size=0.33)

##Declare the Image Generator we will use to augment our training data. 
gen = ImageDataGenerator(horizontal_flip=True)
gen.fit(x_train)

my_model = model.my_model()

##Set up our loss function and call backs
mcp_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')

callbacks = [mcp_save, reduce_lr_loss]
bs = 10
step_size = len(x_train)/bs

my_model.fit_generator(gen.flow(x_train, y_train, bs), epochs=epochs,
                       callbacks=callbacks,
                       steps_per_epoch=step_size
                       ,validation_data = (x_test, y_test), 
                    verbose=1)

##Evaluate

model.load_weights('mdl_wts.hdf5')
score = model.evaluate(x_train, y_train, verbose=2)


##Prepare our submission to kaggle
df_eval = pd.read_json('input/test.json')

train_scaled = util.get_scaled_imgs(df_train)

df_eval.inc_angle = df_eval.inc_angle.replace('na',0)
train_scaled = train_scaled[np.where(df_train.inc_angle>0)[0]]


util.submit(model, train_scaled,"submissions/v1.csv")

