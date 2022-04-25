# -*- coding: utf-8 -*-
"""
Created on 2021-2022

@author: Rob
"""

import glob
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Geod
from sklearn.utils import shuffle


"""
    1. DATA SELECTION
"""

""" 
 RangeIndex: 9992389 entries, 0 to 9992388
Data columns (total 26 columns):
 #   Column                          Dtype  
---  ------                          -----  
 0   # Timestamp                     object 
 1   Type of mobile                  object 
 2   MMSI                            int64  
 3   Latitude                        float64
 4   Longitude                       float64
 5   Navigational status             object 
 6   ROT                             float64
 7   SOG                             float64
 8   COG                             float64
 9   Heading                         float64
 10  IMO                             object 
 11  Callsign                        object 
 12  Name                            object 
 13  Ship type                       object 
 14  Cargo type                      object 
 15  Width                           float64
 16  Length                          float64
 17  Type of position fixing device  object 
 18  Draught                         float64
 19  Destination                     object 
 20  ETA                             object 
 21  Data source type                object 
 22  A                               float64
 23  B                               float64
 24  C                               float64
 25  D                               float64
"""

# Path to the data (folders)
path = ['data']

frames = []
for p in path:
    # Get data file names
    all_files = glob.glob(p + "/*.csv")
    for filename in all_files:
        df = pd.read_csv(filename, header=0, sep=',', index_col=None)
        frames.append(df)

# Concatenate all data into one DataFrame
df = pd.concat(frames)

# Remove temp variables from workspace
del path, all_files, frames, filename, p


"""
    2. REGION FILTER
"""

# boundary box
westbc = 12.00
eastbc = 15.00
northbc = 56.00
southbc = 54.00

# filter by region
data = df.loc[df['Longitude'] >= westbc]
data = data.loc[data['Longitude'] <= eastbc]
data = data.loc[data['Latitude'] >= southbc]
data = data.loc[data['Latitude'] <= northbc]

# filter by vessel type
data = data.loc[data['Ship type'] == 'Cargo']

# Remove temp variables from workspace
del westbc, eastbc, northbc, southbc, df


"""
    2. DATA TRANSFORMATIONS / AUGMENTATIONS
"""

# Calculate date difference in minutes (time steps)
data['DateTime'] = pd.to_datetime(data['# Timestamp'])

dfInterpolator = list()
vessels = data["MMSI"].unique()
for vessel in vessels:
    temp = data.loc[data['MMSI'] == vessel]
    temp = temp.set_index(pd.DatetimeIndex(temp['DateTime']))
    temp = temp.drop_duplicates(subset=['DateTime'])
    temp = temp.resample('60s').nearest(limit=1).dropna(how='all')
    temp['# Timestamp'] = temp.index
    dfInterpolator.append(temp.reset_index(drop=True))
    
dfInterpolator = pd.concat(dfInterpolator)
dfInterpolator = dfInterpolator.sort_values(["MMSI", "DateTime"], ascending = (True, True))
data = dfInterpolator

# Calculate date difference between current and next date
data['DateDiff'] = data['DateTime'].shift(-1) - data['DateTime']
data['DateDiff'] = data['DateDiff'].dt.total_seconds().fillna(0).astype(int) / 60

# Calculate delta latitude, delta longitude
data["dLat/dt"] = ((data['Latitude'].shift(-1) - data['Latitude']) / (data['DateDiff']/60)).fillna(0)
data["dLong/dt"] = ((data['Longitude'].shift(-1) - data['Longitude']) / (data['DateDiff']/60)).fillna(0)

data["Latitude_next"] = data['Latitude'].shift(-1)
data["Longitude_next"] = data['Longitude'].shift(-1)
geod = Geod(ellps='WGS84', proj="utm", zone=33)
for index, row in data.iterrows():
    azimuth_f, azimuth_b, distance = geod.inv(row['Longitude'], row['Latitude'], row['Longitude_next'], row['Latitude_next'])
    data.loc[index,'distance'] = distance


# Remove temp variables from workspace
del temp, vessels, vessel


"""
    3. STATISTICS
"""

# select interested features
data = data[["MMSI", "DateTime", "# Timestamp", "Latitude", "Longitude", "SOG", "Heading", "DateDiff", "dLat/dt", "dLong/dt"]]

data = data.loc[data['SOG'] != 0] # (speed = 0)
data.isnull().sum().sum()   # (nan values)
data = data.dropna()
data = data.drop_duplicates()

data = data.loc[data['DateDiff'] > 0]
data = data.loc[data['DateDiff'] <= 120]
data = data.loc[data['distance'] > 0]
data = data.loc[data['distance'] <= 800]

# data info
data.head()
data.describe()
data.info()
data.shape

np.median(data['DateDiff'])
statistics.mode(data['DateDiff'])
data['DateDiff'].min()
data['DateDiff'].max()

# Basic plot(s)
plt.figure(figsize=(5,5))
data.hist(bins=50, figsize=(20,15))
plt.show()

# Correlation
corr_matrix = data.corr()
plt.figure(figsize=(11,8))
sns.heatmap(corr_matrix, cmap="Greens",annot=True)
plt.show()

# Remove temp variables from workspace
del corr_matrix


"""
    4. SEQUENCING
"""

# main sequence variables
seq_input_length = 30;
seq_output_length = 20;
seq_length = seq_input_length+seq_output_length
window_size = seq_length // 2

# [1,2,3,4,5,6,7,8,9] sequence example if seq length 4 with 2 steps window size
# [1,2,3,4]
# [3,4,5,6]
# [5,6,7,8]
# . . .

seq = []
for i in range(0, len(data) - seq_length + 1, window_size):
    temp = data[i: i + seq_length].values
    # check if all sequences hold only one MMSI (first == last)
    if(temp[0,0] == temp[-1,0]):
        seq.append(temp)
        
# transformation
seq = np.dstack(seq)
seq = np.rollaxis(seq,-1)

# mix the data
seq = shuffle(seq)

np.save("sequences.npy", seq)