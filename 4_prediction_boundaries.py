# -*- coding: utf-8 -*-
"""
Created on 2023-2024

@author: Rob
"""

import numpy as np
from scipy.stats import t
from scipy.spatial import ConvexHull
from scipy.stats import chi2
from shapely.geometry import Point, Polygon


"""
    CALCULATE THE WIDTH OF THE PREDICTION/CONFIDENCE INTERVAL FOR A GIVEN DATASET
"""
def prediction_interval_width(data, confidence=0.95):
    n = len(data)
    std_dev = np.std(data, axis=0, ddof=1)  # Use ddof=1 for sample standard deviation
    t_score = t.ppf((1 + confidence) / 2, n - 1)
    width = t_score * std_dev * np.sqrt(1 + 1/n) * 1  # Multiply by 2 for the full width
    return width

def confidence_interval_width(data, confidence=0.95):
    n = len(data)
    std_dev = np.std(data, axis=0, ddof=1)  # Use ddof=1 for sample standard deviation
    se = std_dev / np.sqrt(n)  # Standard error
    t_score = t.ppf((1 + confidence) / 2, n - 1)  # Corrected: degrees of freedom as positional argument
    return t_score * se * 1  # Multiply by 2 for the full width


"""
    EPR
"""
def calculate_epr(points, center):
    # Compute the Convex Hull of the points
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    
    # Compute eigenvalues and eigenvectors of the Convex Hull vertices
    eig_val, eig_vec = np.linalg.eigh(np.cov(hull_vertices.T))
    
    # Set the confidence level for the EPR
    confidence_level = 0.95
    
    # Compute the radii of the EPR ellipsoid
    radii = np.sqrt(chi2.ppf(confidence_level, 2)) * np.sqrt(eig_val)
    
    # Generate an array of angles from 0 to 2π
    angles = np.linspace(0, 2 * np.pi, 100)
    
    # Compute points on the surface of the ellipsoid
    ellipsoid_points = np.vstack((radii[0] * np.cos(angles), radii[1] * np.sin(angles)))
    
    # Rotate the points by the eigenvectors
    ellipsoid_points = eig_vec.dot(ellipsoid_points).T
    
    # Create the EPR polygon
    epr = Polygon(ellipsoid_points + center)
    
    return epr


"""
    CPR
"""

yhat, ytrue = []
new_prediction, new_actual = []

# Adjust the function to work with the given structure of the calibration data
def calculate_euclidean_distances(predictions, actuals):
    # Calculate Euclidean distances for each step in each sample
    distances = np.sqrt(np.sum((predictions - actuals) ** 2, axis=-1))
    return distances

# Compute nonconformity scores across all samples and steps
nonconformity_scores = calculate_euclidean_distances(yhat.astype(float), ytrue[:,:,1:3].astype(float))

alpha = 0.05
# Since we're interested in stepwise thresholds, calculate them across samples for each step
thresholds_stepwise = np.quantile(nonconformity_scores, 1 - alpha, axis=0)
# thresholds_stepwise = [0.001160502, 0.002049899, 0.003152073, 0.004425105, 0.005825049, 0.007313641, 0.008895545, 0.010564776, 0.01228731, 0.014062625, 0.015865445, 0.017705635, 0.01959625, 0.02153882, 0.023542455, 0.02554966, 0.02761748, 0.029719345, 0.031865835, 0.034013125]

# Now compute nonconformity scores for the new prediction against the new actual
new_scores = calculate_euclidean_distances(new_prediction.reshape(1, 4, 2), new_actual.reshape(1, 4, 2)).flatten()

# Comparing new prediction scores against the thresholds for each step
is_within_threshold = new_scores <= thresholds_stepwise

# Output the thresholds and whether the new scores are within these thresholds for each step
thresholds_stepwise, is_within_threshold, new_scores


"""
    PROBABILITY
"""
def calculate_overlap(epr1, epr2):
    intersection = epr1.intersection(epr2)
    #print(epr1.area, epr2.area, intersection.area)
    #print((intersection.area / epr1.area) * 100, (intersection.area / epr2.area) * 100)
    overlap_percentage = intersection.area / epr1.area * 100
    return overlap_percentage