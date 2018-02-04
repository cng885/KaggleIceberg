# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:56:14 2017

@author: Chase Ginther
"""


import numpy as np 
import pandas as pd 
import os
import cv2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab

#keras imports

from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

#plt.rcParams['figure.figsize'] = 10, 10

train = pd.read_json('input/train.json')

def get_scaled_imgs(df):
    """
    basic function for reshaping and rescaling data as images
    """
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)    


##Function to flip images to create more training data.  
def get_more_images(imgs):
    """
    augmentation for more data
    """    


    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h))
    
    return more_images


Xtrain = get_scaled_imgs(train)
Ytrain = np.array(train['is_iceberg'])
train.inc_angle = train.inc_angle.replace('na',0)
idx_tr = np.where(train.inc_angle>0)

Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

Xtr_more = get_more_images(Xtrain) 
Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))

##Take a look at a iceberg
#import plotly.offline as py
#import plotly.graph_objs as go
#py.init_notebook_mode(connected=True)
#
#def plotmy3d(c, name):
#
#    data = [
#        go.Surface(
#            z=c
#        )
#    ]
#    layout = go.Layout(
#        title=name,
#        autosize=False,
#        width=700,
#        height=700,
#        margin=dict(
#            l=65,
#            r=50,
#            b=65,
#            t=90
#        )
#    )
#    fig = go.Figure(data=data, layout=layout)
#    py.iplot(fig)
#
#plotmy3d(X_band_1[12,:,:], 'iceberg')
#plotmy3d(X_band_1[14,:,:], 'Ship')

#define our model

def getModel():
    #Building the model
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    #gmodel.add(Dropout(0.2))

    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #gmodel.add(Dropout(0.2))

    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #gmodel.add(Dropout(0.2))

    #Conv Layer 4
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #gmodel.add(Dropout(0.2))

    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())

    #Dense Layers
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    #Dense Layer 2
    gmodel.add(Dense(256))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))
    
    #Sigmoid Layer
    gmodel.add(Dense(1))
    gmodel.add(Activation('sigmoid'))

    myoptim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=myoptim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel



seed = np.random.seed(20180125)

# K fold CV training
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
for fold_n, (train, test) in enumerate(kfold.split(Xtr_more, Ytr_more)):
    print("FOLD nr: ", fold_n)
    model = getModel()
    
    MODEL_FILE = 'mdl_simple_k{}_wght.hdf5'.format(fold_n)
    batch_size = 32
    mcp_save = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                                       verbose=1, epsilon=1e-4, mode='min')

    # set the epochs to 30 before training on your GPU
    model.fit(Xtr_more[train], Ytr_more[train],
        batch_size=batch_size,
        epochs=1,
        verbose=1,
        validation_data=(Xtr_more[test], Ytr_more[test]),
        callbacks=[mcp_save, reduce_lr_loss])
    
    model.load_weights(filepath = MODEL_FILE)

    score = model.evaluate(Xtr_more[test], Ytr_more[test], verbose=1)
    print('\n Val score:', score[0])
    print('\n Val accuracy:', score[1])

    SUBMISSION = 'result/sub_simple_v1_{}.csv'.format(fold_n)

    df_test = pd.read_json('input/test.json')
    df_test.inc_angle = df_test.inc_angle.replace('na',0)
    Xtest = (get_scaled_imgs(df_test))
    pred_test = model.predict(Xtest)

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
    print(submission.head(10))

    submission.to_csv(SUBMISSION, index=False)
    print("submission saved")


#Ensemble the results

stacked_1 = pd.read_csv('result/sub_simple_v1_0.csv')
stacked_2 = pd.read_csv('result/sub_simple_v1_1.csv')
stacked_3 = pd.read_csv('result/sub_simple_v1_2.csv')
stacked_4 = pd.read_csv('result/sub_simple_v1_3.csv')
stacked_5 = pd.read_csv('result/sub_simple_v1_4.csv')
stacked_6 = pd.read_csv('result/sub_simple_v1_5.csv')
stacked_7 = pd.read_csv('result/sub_simple_v1_6.csv')
stacked_8 = pd.read_csv('result/sub_simple_v1_7.csv')
stacked_9 = pd.read_csv('result/sub_simple_v1_8.csv')
stacked_10 = pd.read_csv('result/sub_simple_v1_9.csv')
    
#join our results.
stacked_all = stacked_1.merge(stacked_2, on='id')
stacked_all = stacked_all.merge(stacked_3, on='id')
stacked_all = stacked_all.merge(stacked_4, on='id')
stacked_all = stacked_all.merge(stacked_5, on='id')
stacked_all = stacked_all.merge(stacked_6, on='id')
stacked_all = stacked_all.merge(stacked_7, on='id')
stacked_all = stacked_all.merge(stacked_8, on='id')
stacked_all = stacked_all.merge(stacked_9, on='id')
stacked_all = stacked_all.merge(stacked_10, on='id')

stacked_all.columns = ['id', 'is_iceberg_1', 'is_iceberg_2', 'is_iceberg_3',
                       'is_iceberg_4', 'is_iceberg_5', 'is_iceberg_6', 'is_iceberg_7',
                       'is_iceberg_8', 'is_iceberg_9', 'is_iceberg_10']

bruh = stacked_all.corr()

stacked_all['is_iceberg_median'] = stacked_all.iloc[:, 1:11].median(axis=1)
stacked_all['is_iceberg_mean'] = stacked_all.iloc[:, 1:11].mean(axis=1)
stacked_all['is_iceberg_max'] = stacked_all.iloc[:, 1:11].max(axis=1)
stacked_all['is_iceberg_min'] = stacked_all.iloc[:, 1:11].min(axis=1)

mean_stack = stacked_all[['id', 'is_iceberg_mean']]
mean_stack.columns = ['id', 'is_iceberg']
mean_stack.to_csv("result/mean_stack_sub.csv", index=False)

median_stack = stacked_all[['id', 'is_iceberg_median']]
median_stack.columns = ['id', 'is_iceberg']
median_stack.to_csv("result/median_stack_sub.csv", index=False)

#target_train=train['is_iceberg']
#X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(Xtr_more, Ytr_more,
#                                                            random_state=1, train_size=0.75)
#
#
#
#
#
#gmodel=getModel()
#gmodel.fit(X_train_cv, y_train_cv,
#          batch_size=24,
#          epochs=10,
#          verbose=1,
#          validation_data=(X_valid, y_valid),
#          callbacks=callbacks)
#
#gmodel.load_weights(filepath=file_path)
#score = gmodel.evaluate(X_valid, y_valid, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#
#
##load our test data. 
#
#test = pd.read_json('input/test.json')
#
#X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
#X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
#X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
#                          , X_band_test_2[:, :, :, np.newaxis]
#                         ,((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)
#predicted_test=gmodel.predict_proba(X_test)
#
#submission = pd.DataFrame()
#submission['id']=test['id']
#submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
#submission.to_csv('sub.csv', index=False)