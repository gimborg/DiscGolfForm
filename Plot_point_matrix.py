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


csv_file = 'point_matrix.csv'
#point_matrix = []

csv = pd.read_csv(csv_file, sep=' ')

#print(csv)

print(csv[1,:])

#csvfile = list(csv.reader(open(csv_file)))

#print (csvfile)


#with open(csv_file, newline='') as csvfile:
#    file_reader = csv.reader(csvfile, delimiter=' ')
#    row_data = {}
#    for row in xrange():
#        point_matrix.append(row)        

#print(point_matrix)



#with open(csv_file, newline='') as f:
#    reader = csv.reader(f)
#    w = []
#    for row in reader:
#        w.extend(row)

#print (w)

#csvfile = list(csv.reader(open('data.csv')))

#csvdics = []

#for row in csvfile:
#    row_dict = {}
#    for i in xrange(len(row)):
#        row_dict['column_%s' % i] = row[i]
#    csvdics.append(row_dict)