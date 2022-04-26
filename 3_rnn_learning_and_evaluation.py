# -*- coding: utf-8 -*-
"""
Created on 2021-2022

@author: Rob
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pyproj import Proj, Geod
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Dropout, LSTM, Input, RepeatVector, GRU, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from maeh_extension import MAEH_Extension as ext


"""
    LOADING DATA AND PREPARE STRUCTURE (FLAT)
"""
# load preprocessed data by experiments
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
y_test = np.load("y_test.npy")


# depends by experiment
X_train = X_train[:,:,:5] # drop d/Lat and d/Lon - FOR EXP-A ONLY
X_val = X_val[:,:,:5] # drop d/Lat and d/Lon - FOR EXP-A ONLY
X_test = X_test[:,:,:5] # drop d/Lat and d/Lon - FOR EXP-A ONLY
y_train = y_train[:,:,:2] # drop all features except lat and long
y_val = y_val[:,:,:2] # drop all features except lat and long
y_test = y_test[:,:,:2] # drop all features except lat and long


# flatten output structure
y_train_flat = np.copy(y_train)
y_val_flat = np.copy(y_val)
y_test_flat = np.copy(y_test)
y_train_flat = y_train_flat.reshape(y_train_flat.shape[0], y_train_flat.shape[1]*y_train_flat.shape[2])
y_val_flat = y_val_flat.reshape(y_val_flat.shape[0], y_val_flat.shape[1]*y_val_flat.shape[2])
y_test_flat = y_test_flat.reshape(y_test_flat.shape[0], y_test_flat.shape[1]*y_test_flat.shape[2])


"""
    HYPERPARAMETERS
"""
geod = Geod(ellps='WGS84', proj="utm", zone=33)
p = Proj (
    proj="utm",
    zone=33,
    ellps="WGS84"
)

# experiment parameters
lstm_cells = [25,50,75,100,125,150,175,200,225,250,275,300]
model_repeat_time = 10

seq_input_length = 30;
seq_output_length = 20;

# network parameters
n_features = X_train.shape[2]
lstm_layer_size = 25
num_epochs = 100
batch_size = 128


"""
    DEEP LEARNING ARCHITECTURES
"""
# define LSTM AE model (Sequence-to-sequence autoencoder)
def lstm_autoencoder_flat(rewrite_model, index):
    model_file_name = f'LSTM-AE_cells-{lstm_layer_size}_epochs-{num_epochs}_batch-{batch_size}_Test-{index}.h5'
    history_file_name = f'LSTM-AE_cells-{lstm_layer_size}_epochs-{num_epochs}_batch-{batch_size}_Test-{index}_history.npy'
    if(os.path.isfile(model_file_name) and not rewrite_model):
        autoencoder = load_model(model_file_name)
        history = np.load(history_file_name, allow_pickle=True).item()
    else:
        # define encoder
        inputs = Input(shape=(seq_input_length, n_features))
        encoder = LSTM(lstm_layer_size, activation='relu', return_sequences=False)(inputs)
        encoder = Dropout(0.01)(encoder)
        # define reconstruct decoder
        decoder = RepeatVector(seq_output_length)(encoder)
        decoder = LSTM(lstm_layer_size, activation='relu', return_sequences=False)(decoder)
        decoder = Dropout(0.01)(decoder)
        decoder = Dense(y_train_flat.shape[1], activation='relu')(decoder)
        # create autoencoder
        if(os.path.isfile(model_file_name)):
           autoencoder = load_model(model_file_name)
        else:
            autoencoder = Model(inputs, decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=8, min_lr=0.0001)
        earlyStop = EarlyStopping(monitor="val_loss", verbose=1, mode='min', patience=8)
        checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # fit model
        history = autoencoder.fit(X_train, y_train_flat, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val_flat),
                                  callbacks=[reduce_lr,earlyStop,checkpoint])
        np.save(history_file_name, history.history)
    return autoencoder, history


def basic_gru_flat(rewrite_model, index):
    model_file_name = f'Basic-GRU_cells-{lstm_layer_size}_epochs-{num_epochs}_batch-{batch_size}_Test-{index}.h5'
    history_file_name = f'Basic-GRU_cells-{lstm_layer_size}_epochs-{num_epochs}_batch-{batch_size}_Test-{index}_history.npy'
    model = Sequential()
    if(os.path.isfile(model_file_name) and not rewrite_model):
        model = load_model(model_file_name)
        history = np.load(history_file_name, allow_pickle=True).item()
    else:
        if(os.path.isfile(model_file_name)):
           model = load_model(model_file_name)
        else:
            model = Sequential()
            
        model.add(GRU(lstm_layer_size, activation='relu', input_shape=(seq_input_length, n_features)))
        model.add(Dropout(0.01))
        
        model.add(Dense(y_train_flat.shape[1], activation='relu'))
        model.compile(optimizer='adam', loss='mse')
        
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=8, min_lr=0.0001)
        earlyStop = EarlyStopping(monitor="val_loss", verbose=1, mode='min', patience=8)
        checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # fit model
        history = model.fit(X_train, y_train_flat, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val_flat), 
                            callbacks=[reduce_lr,earlyStop,checkpoint])
        np.save(history_file_name, history.history)
    return model, history


def basic_bidirectional_lstm_flat(rewrite_model, index):
    model_file_name = f'Basic-Bidirectional-LSTM_cells-{lstm_layer_size}_epochs-{num_epochs}_batch-{batch_size}_Test-{index}.h5'
    history_file_name = f'Basic-Bidirectional-LSTM_cells-{lstm_layer_size}_epochs-{num_epochs}_batch-{batch_size}_Test-{index}_history.npy'
    model = Sequential()
    if(os.path.isfile(model_file_name) and not rewrite_model):
        model = load_model(model_file_name)
        history = np.load(history_file_name, allow_pickle=True).item()
    else:
        if(os.path.isfile(model_file_name)):
           model = load_model(model_file_name)
        else:
            model = Sequential()
            
        model.add(Bidirectional(LSTM(lstm_layer_size, activation='relu'), input_shape=(seq_input_length, n_features)))
        model.add(Dropout(0.01))
        
        model.add(Dense(y_train_flat.shape[1], activation='relu'))
        model.compile(optimizer='adam', loss='mse')
        
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=8, min_lr=0.0001)
        earlyStop = EarlyStopping(monitor="val_loss", verbose=1, mode='min', patience=8)
        checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # fit model
        history = model.fit(X_train, y_train_flat, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val_flat), 
                            callbacks=[reduce_lr,earlyStop,checkpoint])
        np.save(history_file_name, history.history)
    return model, history


"""
    MODELS TRAINING
"""
def write_logs(modelName):
    end_time = datetime.now()
    duration = end_time - begin_time
    m,s = divmod(duration.total_seconds(), 60)
    f = open('logs.txt', "a")
    f.write(modelName + '\n')
    f.write('Begin datetime: {} \t\t End datetime: {}'.format(begin_time, end_time) + '\n')
    f.write('Duration H:M:S is {}:{}:{}'.format(m//60,m%60,s) + '\n')
    model.summary(print_fn=lambda log: f.write(log + '\n'))
    f.write(model.to_json() + '\n\n')
    f.close()
    
    
# training by cell size
for lstm_layer_size in lstm_cells:
    for i in range(model_repeat_time):
        begin_time = datetime.now()
        model, history = lstm_autoencoder_flat(True, i) # training Autoencoder models
        # model, history = basic_gru_flat(True, i)  # training GRU models
        # model, history = basic_bidirectional_lstm_flat(True, i)  # training LSTM Bidirectional models
        write_logs('LSTM-AE_cells-{}_epochs-{}_batch-{}_Test-{}'.format(lstm_layer_size, num_epochs, batch_size, i))


"""
    ADDITIONAL METHODS
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

def transform_and_scale_utm(x):
    i = x.shape[1]
    x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
    x = scaler_utm.inverse_transform(x)
    x = x.reshape(x.shape[0]//i, i, x.shape[1])
    return x

def inverse_pol_to_coordinates(x, y_pred):
    x = transform_and_scale(x)
    lat = x[:,seq_input_length-1,0]
    long = x[:,seq_input_length-1,1]
    
    prediction = list()
    for j in range(len(x)):
        temp_lat = lat[j]
        temp_long = long[j]
        for i in range(seq_output_length):
            endLon,endLat,backAzimuth = geod.fwd(temp_long,temp_lat,y_pred[j,i,0],y_pred[j,i,1])
            prediction.append([endLat, endLon])
            temp_lat = endLat
            temp_long = endLon
    prediction = np.array(prediction)
    prediction = prediction.reshape(len(prediction)//seq_output_length,seq_output_length,2)
    return prediction

def inverse_utm_to_coordinates(X_val, y_pred):
    X_val = transform_and_scale(X_val)
    lat = X_val[:,seq_input_length-1,0]
    long = X_val[:,seq_input_length-1,1]
    
    y_pred = y_pred.reshape(len(y_pred)*seq_output_length, 2)
    y_pred = scaler_utm.inverse_transform(y_pred)
    y_pred = y_pred.reshape(len(y_pred)//seq_output_length,seq_output_length, 2)
    
    prediction = list()
    
    for j in range(len(X_val)):
        temp_lat = lat[j]
        temp_long = long[j]
        for i in range(seq_output_length):
            utm_x, utm_y = p(temp_lat, temp_long, inverse=False)
            temp_x = utm_x + y_pred[j,i,0]
            temp_y = utm_y + y_pred[j,i,1]
            
            endLat, endLon = p(temp_x, temp_y, inverse=True)
            prediction.append([endLat, endLon])
            temp_lat = endLat
            temp_long = endLon
    prediction = np.array(prediction)
    prediction = prediction.reshape(len(prediction)//seq_output_length,seq_output_length,2)
    
    return prediction

def plt_dynamic_training(x, vy, ty, ax, title, colors=['b']):
    ax.plot(x, vy, 'r', label="Validation Loss")
    ax.plot(x, ty, 'b', label="Train Loss")
    ax.legend()
    plt.title(title)
    plt.grid()
    plt.savefig(f'image_{lstm_layer_size}_{i}.jpg')
    fig.canvas.draw()
    

"""
    MODELS EVALUATION
"""
scaler = pickle.load(open("scaler.pkl", "rb"))
scaler_pol = pickle.load(open("scaler_pol.pkl", "rb"))
scaler_utm = pickle.load(open("scaler_utm.pkl", "rb"))

ytrue = np.load("y_test.npy")
ytrue = transform_and_scale(ytrue)

loss_avg = list()
distance_loss = np.zeros((model_repeat_time, len(lstm_cells)))
ii = 0
for lstm_layer_size in lstm_cells:
    score = 0
    for i in range(model_repeat_time):
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAEH (Loss)')
        # load model and history without training new model
        model, history = lstm_autoencoder_flat(False, i)
        # list of each epoch numbers
        x = list(range(1, len(history['val_loss'])+1))
        # model evaluation (forecast/prediction)
        yhat = model.predict(X_test, verbose=0, batch_size=batch_size)
        
        """FOR LAT LON"""
        # EXP-A and EXP-B
        yhat = yhat.reshape(len(yhat), seq_output_length, 2)
        yhat = transform_and_scale(yhat)

        """FOR POL"""
        # EXP-C
        """yhat = yhat.reshape(len(yhat), seq_output_length, 2) #flat
        yhat = yhat.reshape(len(yhat)*seq_output_length, 2)
        yhat = scaler_pol.inverse_transform(yhat)
        yhat = yhat.reshape(len(yhat)//seq_output_length,seq_output_length, 2)
        yhat = inverse_pol_to_coordinates(X_test, yhat)"""
        
        """FOR UTM"""
        # EXP-D
        """yhat = inverse_utm_to_coordinates(X_test,yhat)"""

        error, dist = ext.true_and_predicted_distance(ext, ytrue, yhat, 'k')
        score += error
        distance_loss[i,ii] = error
        
        # training plot
        vy = history['val_loss']
        ty = history['loss']
        plt_dynamic_training(x[3:], vy[3:], ty[3:], ax, f'Cross Validation - AE-LSTM-Exp_A_cells-{lstm_layer_size}_Experiment-{i}')
    ii = ii + 1
    loss_avg.append(score/model_repeat_time)
    print('AVERAGE TEST SCORE: ', score/model_repeat_time)