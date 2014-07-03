import cv2
import numpy as np
def pickle_keypoints(keypoints, descriptors):  #Change format of kp1 returned by sift
    i=0
    temp_array = []
    for point in keypoints:
        temp = descriptors[i]
        i+=1
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):  #Change format of file to Numpy Array
    keypoints = []
    descriptors = []
    for point in array:      
        #temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        #keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return descriptors
