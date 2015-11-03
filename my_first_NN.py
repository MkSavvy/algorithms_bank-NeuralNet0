# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 08:58:48 2015

@author: Michel_home
"""
from __future__ import division
import numpy as np


# define sigmoid and derivative of sigmoid
def sigmoid(X, slope = False):
    if slope:
        return X*(1-X)
    return 1/(1+np.exp(X))

# initialize inputs and labels (dependent variable)
X = np.array([[1,1,0],[1,0,1],[0,0,1],[0,0,0]]) # 4x3 matrix
Y = np.array([1,1,0,0]).T # a column vector of length = # events

# initialize a random weight for each 'p' node (each predictor column)
weights = 2*np.random.random((3,1)) - 1

# start looping over each datapoint
for iter in xrange(1):
   # randomly select a portion of the input vectors as current input


   # compute the value after each node by dot product
   # into l1
    l1 = np.dot(X.T, weights)
    
   # Does weighted sum pass the sigmoid?
    delta = sigmoid(l1) # the prediction of the activation
    
   # check difference
    l1_error = Y - delta
    
    # multiply the difference by the slope of the sigmoid
    correction = np.dot(l1_error, sigmoid(X, True))
    
    # must update weights with slope
    weights = weights + correction