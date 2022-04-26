# -*- coding: utf-8 -*-
"""
Created on 2021-2022

@author: Rob
"""

import glob
import pickle
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj, Geod

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# main sequence variables
seq_input_length = 30;
seq_output_length = 20;
seq_length = seq_input_length+seq_output_length

"""
    5. LATITUDE/LONGITUDE NORMALIZATION AND DATA SPLITTING (TRAIN, VALIDATION, TEST) (EXP-A, EXP-B)
"""
seq = np.load("sequences.npy", allow_pickle=True)
seq = seq.reshape(seq.shape[0]*seq.shape[1], seq.shape[2])
seq = seq[:,3:] # drop MMSI and DateTime

# train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(seq)

# normalize the dataset and print
seq = scaler.transform(seq)
print('Min: ' + ',\t'.join(map(str,scaler.data_min_)) + '\nMax: ' + ',\t'.join(map(str,scaler.data_max_)))

seq = seq.reshape(seq.shape[0]//seq_length, seq_length, seq.shape[1])

X = seq[:,0:seq_input_length,:]
y = seq[:,seq_input_length:seq_length,:]

# save/load scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))
# scaler = pickle.load(open("scaler.pkl", "rb"))

# split train and test data by 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50)   # 70 : 15 : 15

"""
# save preprocessed data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("y_val.npy", y_val)
"""

def transform_and_scale(x):
    i = 7 - x.shape[2]
    if(i != 0):
        x = np.append(x, np.zeros((x.shape[0], x.shape[1], i)), axis=2)
    i = x.shape[1]
    x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
    x = scaler.inverse_transform(x)
    x = x.reshape(x.shape[0]//i, i, x.shape[1])
    return x


"""
    6. POLAR STRATEGY and NORMALIZATION (EXP-C)
"""
geod = Geod(ellps='WGS84', proj="utm", zone=33)

def convert_coord_to_distance_and_angle(x, y):
    x = transform_and_scale(x)
    y = transform_and_scale(y)
    
    yyy = list()    
    for j in range(len(y)):
        xx = x[j]
        yy = y[j]
        
        temp = np.copy(yy)
        temp = np.insert(temp, 0, xx[seq_input_length-1], axis=0)
        
        new_y = np.zeros((seq_output_length,2))
        
        for i in range(len(yy)):
            currentLat, currentLon = temp[i,0], temp[i,1]
            nextLat, nextLon = yy[i,0], yy[i,1]
            azimuth_f, azimuth_b, distance = geod.inv(currentLon, currentLat, nextLon, nextLat)
            new_y[i,0] = azimuth_f
            new_y[i,1] = distance
        yyy.append(new_y)
    return np.array(yyy)


y_train = convert_coord_to_distance_and_angle(X_train, y_train)
y_test = convert_coord_to_distance_and_angle(X_test, y_test)
y_val = convert_coord_to_distance_and_angle(X_val, y_val)

df_y_train = pd.DataFrame(y_train.reshape(y_train.shape[0]*y_train.shape[1], y_train.shape[2]), columns=["angle", "dist"])
df_y_test = pd.DataFrame(y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2]), columns=["angle", "dist"])
df_y_val = pd.DataFrame(y_val.reshape(y_val.shape[0]*y_val.shape[1], y_val.shape[2]), columns=["angle", "dist"])
df = pd.concat([df_y_train,df_y_test,df_y_val])
df = df[["angle", "dist"]]

# train the normalization
scaler_pol = MinMaxScaler(feature_range=(0, 1))
scaler_pol = scaler_pol.fit(df)
print('Min: ' + ',\t'.join(map(str,scaler_pol.data_min_)) + '\nMax: ' + ',\t'.join(map(str,scaler_pol.data_max_)))

# normalize the dataset
def polar_norm(x):
    x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
    x = scaler_pol.transform(x)
    x = x.reshape(x.shape[0]//seq_output_length, seq_output_length, x.shape[1])
    return x

y_train = polar_norm(y_train)
y_test = polar_norm(y_test)
y_val = polar_norm(y_val)

# save/load scaler
pickle.dump(scaler_pol, open("scaler_pol.pkl", "wb"))
# scaler_pol = pickle.load(open("scaler_pol.pkl", "rb"))

"""
# save preprocessed data
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("y_val.npy", y_val)
"""

"""
    7. UTM STRATEGY and NORMALIZATION (EXP-D)
"""
p = Proj (
    proj="utm",
    zone=33,
    ellps="WGS84"
)

def convert_coord_to_utm(x, y):    
    x = transform_and_scale(x)
    y = transform_and_scale(y)
    yyy = list()
    
    for j in range(len(y)):
        xx = x[j]
        yy = y[j]
        
        temp = np.copy(yy)
        temp = np.insert(temp, 0, xx[seq_input_length-1], axis=0)
        
        new_y = np.zeros((seq_output_length,2))
        
        for i in range(len(yy)):
            currentLat, currentLon = p(temp[i,0], temp[i,1], inverse=False)            
            nextLat, nextLon = p(yy[i,0], yy[i,1], inverse=False)
            
            new_y[i,0] = nextLat - currentLat # utm x
            new_y[i,1] = nextLon - currentLon  # utm y
        yyy.append(new_y)
    return np.array(yyy)


y_val = convert_coord_to_utm(X_val,y_val)
y_test = convert_coord_to_utm(X_test,y_test)
y_train = convert_coord_to_utm(X_train,y_train)

df_y_train = pd.DataFrame(y_train.reshape(y_train.shape[0]*y_train.shape[1], y_train.shape[2]), columns=["east", "north"])
df_y_test = pd.DataFrame(y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2]), columns=["east", "north"])
df_y_val = pd.DataFrame(y_val.reshape(y_val.shape[0]*y_val.shape[1], y_val.shape[2]), columns=["east", "north"])
df = pd.concat([df_y_train,df_y_test,df_y_val])
df = df[["east", "north"]]

# train the normalization
scaler_utm = MinMaxScaler(feature_range=(0, 1))
scaler_utm = scaler_utm.fit(df)
print('Min: ' + ',\t'.join(map(str,scaler_utm.data_min_)) + '\nMax: ' + ',\t'.join(map(str,scaler_utm.data_max_)))

# normalize the dataset
def utm_norm(x):
    x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
    x = scaler_utm.transform(x)
    x = x.reshape(x.shape[0]//seq_output_length, seq_output_length, x.shape[1])
    return x

y_train = utm_norm(y_train)
y_test = utm_norm(y_test)
y_val = utm_norm(y_val)

# save/load scaler
pickle.dump(scaler_utm, open("scaler_utm.pkl", "wb"))
# scaler_utm = pickle.load(open("scaler_utm.pkl", "rb"))

"""
# save preprocessed data
np.save("y_train.npy", temp)
np.save("y_test.npy", temp)
np.save("y_val.npy", temp)
"""