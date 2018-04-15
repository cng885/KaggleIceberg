# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 21:14:16 2018

@author: Chase
"""

import numpy as np
import pandas as pd

def get_scaled_imgs(df):
    """
    basic function for reshaping and rescaling data as images
    """
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = (band_1 + band_2)/2 # plus since log(x*y) = log(x) + log(y)
#        
#        # Rescale
#        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
#        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
#        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        #imgs.append(np.dstack((a, b, c)))
        imgs.append(np.dstack((band_1, band_2, band_3)))
        
    return np.array(imgs)    

def submit(model,test_df ,x, out_path):
    
    pred = model.predict(x)
    submission = pd.DataFrame({'id': test_df["id"], 
                               'is_iceberg': pred.reshape((pred.shape[0]))})
    submission.to_csv(out_path, index=False)
