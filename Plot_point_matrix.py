# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:40:13 2024

@author: Jakob
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


csv_file = 'Data/point_matrix.csv'
#point_matrix = []

points_dataframe = pd.read_table(csv_file, sep=' ')


points = pd.DataFrame(points_dataframe).to_numpy()


np.sort(points[:,2])

idx= find(points[:,1]==5)

idx

#print(points[1:3,:])


#people.name  #  all the names
#people.loc[0:2] # first two rows







# with open(csv_file) as file:
#    lines = file.read()

# print(len(lines))


# for cnt in range(1:len(lines)):
#     print (str(lines[i,:]))




#print(lines)
