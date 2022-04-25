# -*- coding: utf-8 -*-
"""
Created on 2021-2022

@author: Rob
"""

import numpy as np
from pyproj import Proj, Geod

class MAEH_Extension:
    
    p, geon = None, None # variables for cartographic transformations
    
    def __init__(self):
        self.p = Proj (
            proj="utm",
            zone=33,
            ellps="WGS84",
            south=True
        )    
        self.geod = Geod(ellps='WGS84', proj="utm", zone=33)
        
# In[1]: ADDITIONAL FUNCTIONS
        
    def haversine_distance(lat1, lon1, lat2, lon2, units):
        if(units == 'k'):
            r = 6371 # Radius of earth in kilometers
        elif(units == 'm'):
            r = 6371000 # Radius of earth in meters
        else:
            r = 3956 # Radius of earth in miles            
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
        res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
        return np.round(res, 2)
    
    def true_and_predicted_distance(self, y, yhat, units):
        # calculate distance between true and predicted points
        if(y.ndim == 2):
            y = y.reshape(1, y.shape[0], y.shape[1])
        if(yhat.ndim == 2):
            yhat = yhat.reshape(1, yhat.shape[0], yhat.shape[1])
        dist = self.haversine_distance(y[:,:,0], y[:,:,1], yhat[:,:,0], yhat[:,:,1], units)
        return np.average(abs(dist)), dist