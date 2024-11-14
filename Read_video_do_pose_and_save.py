# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:06:36 2024

@author: Jakob
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import YouTubeVideo, display, Image

#%matplotlib inline


# def download_and_unzip(url, save_path):
#     print(f"Downloading and extracting assests....", end="")

#     # Downloading zip file using urllib package.
#     urlretrieve(url, save_path)

#     try:
#         # Extracting zip file using the zipfile package.
#         with ZipFile(save_path) as z:
#             # Extract ZIP file contents in the same directory.
#             z.extractall(os.path.split(save_path)[0])

#         print("Done")

#     except Exception as e:
#         print("\nInvalid file.", e)


# #Input parameters #############################################

# URL = r"https://www.dropbox.com/s/089r2yg6aao858l/opencv_bootcamp_assets_NB14.zip?dl=1"

source = 'Data/test.mp4'  # source is the captured video from file
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error opening video stream or file")

property_id = int(cv2.CAP_PROP_FRAME_COUNT)  
length = int(cv2.VideoCapture.get(cap, property_id)) 
print("Total number of frames: " + str(length) )
fps = cap.get(cv2.CAP_PROP_FPS)
print ("Framerate: " + str(fps))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print ("Frame width: " + str(frame_width) + " Frame height: " + str(frame_height))

#out_heatmap = cv2.VideoWriter("Pics_vids/Heat_map_3rd.mp4", cv2.VideoWriter_fourcc(*"XVID"), 10, (frame_width, frame_height))
out_pose    = cv2.VideoWriter("Data/test.mp4", cv2.VideoWriter_fourcc(*"XVID"), 30, (frame_width, frame_height))



#Frame interval
#First shoulders and hips unsynced 154:250
#Second shoulders and hips synced 297:381

#Frm_start_1 = 154
#Frm_stop_1  = 250
#Frm_start_2 = 297
#Frm_stop_2  = 381


# asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_NB14.zip")

# # Download if assest ZIP does not exists. 
# if not os.path.exists(asset_zip_path):
#     download_and_unzip(URL, asset_zip_path)



protoFile   = "Model/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = os.path.join("Model", "pose_iter_160000.caffemodel")


nPoints = 15
POSE_PAIRS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 14],
    [14, 8],
    [8, 9],
    [9, 10],
    [14, 11],
    [11, 12],
    [12, 13],
]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
   
point_matrix = np.zeros((length*15,4))

#point_matrix = np.zeros((8,10))

print(point_matrix)

#for frm in range(1,length):
for frm in range(1,3):
    print("Frame number: " + str(frm))
    ret, frame = cap.read()
    
    #if (frm > Frm_start_1 and frm < Frm_stop_1) or (frm > Frm_start_2 and frm < Frm_stop_2):
    
    
    
    #frame = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2RGB)


    inWidth = frame_width
    inHeight = frame_height


#    inWidth  = frame.shape[1]
#    inHeight = frame.shape[0]
    
    
    netInputSize = (368, 368)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)
    
    # Forward Pass
    output = net.forward()
    
    # Display probability maps
    #plt.figure(figsize=(20, 5))
    #for i in range(nPoints):
    #    probMap = output[0, i, :, :]
    #    displayMap = cv2.resize(probMap, (inWidth, inHeight), cv2.INTER_LINEAR)
    #    
    #    plt.subplot(2, 8, i + 1)
    #    plt.axis("off")
    #    plt.imshow(displayMap, cmap="jet")
    
    
    # X and Y Scale
    scaleX = inWidth  / output.shape[3]
    scaleY = inHeight / output.shape[2]
    
    # Empty list to store the detected keypoints
    points = []
    
    # Treshold
    threshold = 0.1
    
    
    for i in range(nPoints):
        # Obtain probability map
        probMap = output[0, i, :, :]
    
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
        # Scale the point to fit on the original image
        x = scaleX * point[0]
        y = scaleY * point[1]
        
        point_matrix[(frm-1)*15+i,:] = ([frm, i+1, x, y])
        
        
        if prob > threshold:
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)
            
            
    imPoints = frame.copy()
    imSkeleton = frame.copy()
    
    # Draw points
    for i, p in enumerate(points):
        cv2.circle(imPoints, p, 8, (255, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(imPoints, "{}".format(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        
    
    # Draw skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
    
        if points[partA] and points[partB]:
            cv2.line(imSkeleton, points[partA], points[partB], (255, 255, 0), 2)
            cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (20, 30)
    org2 = (20, 40)
    
    # fontScale
    fontScale = 0.50
     
    # Blue color in BGR
    color = (0, 0, 255)
    
    # Line thickness of 2 px
    thickness = 1
     
    # Using cv2.putText() method
    imSkeleton = cv2.putText(imSkeleton, 'Frame: ' + str(frm) + ' / ' + str(length), org, font, fontScale, color, thickness, cv2.LINE_AA)
    imSkeleton = cv2.putText(imSkeleton, 'FPS: ' + str(fps), org2, font, fontScale, color, thickness, cv2.LINE_AA)
    
    
    imSkeleton_RGB = cv2.cvtColor(imSkeleton, cv2.COLOR_RGB2BGR)


    out_pose.write(imSkeleton_RGB)

out_pose.release()



with open('Data/Point_matrix.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ' ')
    my_writer.writerows(point_matrix)

#plt.figure(figsize=(50, 50))

#plt.subplot(121)
#plt.axis("off")
#plt.imshow(imPoints)

#plt.subplot(122)
#plt.axis("off")
#plt.imshow(imSkeleton)
