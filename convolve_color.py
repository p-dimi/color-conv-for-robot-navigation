# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:56:26 2019

@author: Dima's_Monster
"""

import numpy as np
import cv2

########################
### CONVOLUTION PART ###
########################

# convolution kernels
affirm = np.array([[0.1, 0.1, 0.1],
                   [0.1, 1.0, 0.1],
                   [0.1, 0.1, 0.1]])

negate = affirm.copy() * (-1)

# open your image
test_img = cv2.imread('straight.png')

# prep image by scaling it down to 10% its original size for faster processing
test_img_smaller = cv2.resize(test_img, (0,0), fx=0.1, fy=0.1)

# function to add padding to image
def pad(img, kernel):
    # get the necessary padding
    h,w = kernel.shape
    h = h - 1
    w = w - 1
    
    # pad the image
    padded_img = np.zeros((img.shape[0]+(h), img.shape[1]+(w), 3), dtype=np.uint8)
    padded_img[int(h/2) : img.shape[0]+int(h/2), int(w/2) : img.shape[1]+int(w/2), :] = img
        
    return padded_img

# pad the image before performing convolution
test_img_smaller = pad(test_img_smaller, affirm)

# your parameters which decide what color you are looking for (considering opencv's format is BGR)
# so because you're looking for predominantly blue, the index is 0
desired_color_index = 0

# buffer array (this is where each color channel convolution will go, and will be summed later on)
convolved_image = np.zeros(test_img_smaller.shape, dtype=np.float)

# convole over the image
for i in range(test_img_smaller.shape[2]):
    # choose kernel based on which color channel this is
    if i == desired_color_index:
        kernel = affirm
    else:
        kernel = negate
    # convolve
    convolution_outcome = cv2.filter2D(test_img_smaller[:,:,i].astype(np.float), -1, kernel)
    
    # append to buffer array
    convolved_image[:,:,i] = convolution_outcome

# little function to finish up the result (sum up the buffer array, and clamp the values to be between 0 and 255)
def sum_and_clamp(image):
    # sum the buffer array's three channels
    image = image.sum(axis=-1)
    
    # clamp the image's values between 0 and 255
    image = np.where(image < 255.0, image, 255)                
    image = np.where(image > 0.0, image, 0)     
    
    return image

convolved_image = sum_and_clamp(convolved_image)

# display the outcome
cv2.imshow('outcome', convolved_image)


#######################
###  CENTROID PART  ###
#######################

# calculate centroid
M = cv2.moments(convolved_image)
cY = int(M["m01"] / M["m00"])
cX = int(M["m10"] / M["m00"])

# convert centroid coordinates to integer
cY = int(cY)
cX = int(cX)

# draw a white circle on a blank image to demonstrate where the centroid was found
centroid_image = convolved_image.copy() * 0
centroid_image = cv2.circle(centroid_image, (cX,cY) , 3, color=255, thickness=-1, lineType=1)
cv2.imshow('final result', centroid_image)

# Check whether to go straight, turn left, or turn right
# set a certain threshold of how centered the centroid should be (this would be in raw pixels, you will measure this from the center of the X axis of the image)
center_threshold = 5

# measure the distance of the centroid's X value from the X axis center
dist = cX - (convolved_image.shape[1] / 2)

# if it's within the threshold, go straight. otherwise, turn either left or right
if abs(dist) < center_threshold:
    print('Straight')
elif dist < 0:
    print('Left')
elif dist > 0:
    print('Right')