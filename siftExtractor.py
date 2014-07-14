"""
This module extracts the sift descriptors for each image 
stored in ./Train/ folder (Training image Databse). The 
./Train/ folder must have images for particular object in 
repctive object_name sub directories. The extracted descriptors 
are stored as files one for each folder in ./Train/ folder are 
stored in ./Desc/ folder. For consistency they are named according 
to names of the folder in ./Train/ directory.
"""

import os
import cPickle as pickle
import numpy as np
import cv2

def siftExtractor():
	"""
	For each image in sub directory of ./Train/ reads the image 
	and converts it to gray. Grayscale is used to make detection 
	invariant to pixel intensities. It extracts the sift descriptor 
	and stores them in descriptor file in numpy array format. 
	The descriptor are dumped using pickle library in two halves to 
	avoid memory leak while reloading descriptors.
	"""
	path = os.getcwd()

	if not os.path.exists('Desc'):
	    os.makedirs('Desc')

	os.chdir(path + '/'+'Train')
	parentdir = path + '/'+'Train'		#: Parent directory of images folders
	ptlistdir = os.listdir(parentdir)	#: List of folders in parent directory

	sift = cv2.SIFT()		#: intialize sift object
	os.chdir(path)
	for x in range(0,len(ptlistdir)):
	    temp_array=[]		#: Populator for descriptors
	    childdir = parentdir +'/'+ ptlistdir[x]		#: Path of child directory
	    childdir_cpy = childdir		#: Copy of child directory path
	    os.chdir(childdir)
	    cllistdir = os.listdir(childdir)	#: List of images in child directory
	    Keypt = open(path+'/Desc/'+ptlistdir[x]+'.p',"wb")	#: File pointer for descriptor file
	    num = len(cllistdir)/2
	    i = 1

	    print('Directory : '+ptlistdir[x]+' Images : '+str(len(cllistdir)))

	    for y in range(0,len(cllistdir)):
	        childdir_cpy = childdir + '/' + cllistdir[y]
	        print(childdir_cpy)
	        img = cv2.imread(childdir_cpy)		#Read Image
	        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)	#Convert to gray
	        kp, desc = sift.detectAndCompute(gray,None)	#Computing SIFT Keypoints and Descriptors
	        #Populate descriptors one after other. 
	        #extends is used so that there is no distinction or sub array boundaries
	        #between descriptors of images.
	        temp_array.extend(desc)
	        if(i==1 and y>num):	
	        	pickle.dump(np.array(temp_array),Keypt)		#Dump first half to file
	        	temp_array = []
	        	i = 0
	    pickle.dump(np.array(temp_array),Keypt)  #Dump second half to file
	    Keypt.close()
	    del temp_array

if __name__ == '__main__':
	siftExtractor()