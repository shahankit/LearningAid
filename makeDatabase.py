"""
This module creates the database required for SVM. 
It creates a Feature vector for each image in ./Train/
directory and appropriate label vector to label the 
features.
"""

import os
import numpy as np
import cv2
import pickle
import sys

def makeDatabase(histSize):
	"""
	This function finds SIFT descriptor for each image in ./Train/
	directory. It finds the similarity between extracted features 
	and cluster centers from centerFinal.p. It creates a histogram 
	to check number of features mapped to particular cluster center. 
	These are feature vector of images which are appedned and dumped 
	to Database.p file.
	"""
	path = os.getcwd()
	Center = open("centerFinal.p","rb") #: File pointer for centers file
	centers = pickle.load(Center)
	Center.close()
	os.chdir(path + '/'+'Train')
	parentdir = path + '/'+'Train'		#: Parent directory of images folders
	ptlistdir = os.listdir(parentdir)	#: List of folders in parent directory

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) # Create BFMatcher Object
	sift =cv2.SIFT()	# Intialize sift object
	feat_array = []		#: Populator for features
	label = []			#: Labels for images, unique for each object

	for x in range(0,len(ptlistdir)):
	    childdir = parentdir +'/'+ ptlistdir[x]		#: Path of child directory
	    childdir_cpy = childdir 		#: Copy of child directory path
	    os.chdir(childdir)
	    cllistdir = os.listdir(childdir)	#: List of images in child directory
	    print('In '+ptlistdir[x])
	    for y in range(0,len(cllistdir)):
	        label.append(x)		#Append same label for all images in same folder
	        Histogram = [0]*int(histSize)
	        childdir_cpy = childdir + '/' + cllistdir[y]
	        print(childdir_cpy)
	        img = cv2.imread(childdir_cpy)	#Read Image
	        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)	#Convert to gray
	        kp, desc = sift.detectAndCompute(img,None)	#Computing Sift Keypoints and Descriptors
	        matches = bf.match(desc,centers)	#Matching Feature
	        print(childdir_cpy)
	        for i in matches:
	            Histogram[i.trainIdx]+=1	#Calculate histogram
	        feat_array.append(Histogram)
	        del Histogram
	os.chdir(path)
	Center.close()

	Database = open("Database.p","wb")
	pickle.dump(feat_array,Database)	#Dump features to file
	pickle.dump(label,Database)		#Dump labels to file
	Database.close()

if __name__ == '__main__':
    if(len(sys.argv)!=2):
        print('Error. Usage: python makeDatabase.p histogramSize')
        print('Hint : histogramSize is same as number of centers')
    else:
        makeDatabase(sys.argv[1])
