"""
This module takes the descriptor form ./Desc/ directory 
and clusters then into number of specified centers for 
each descriptor file. The cluster centers are saved to 
./Centers/ directory.
"""

import numpy as np
import cv2
from cv2 import __version__
import pickle
import os
import sys

def kMeans(nCenters):
	"""
	This function takes descriptors stored in ./Desc/ 
	directory one file at time. Loads the first half 
	and the runs kmeans from cv2 library to find out 
	cluster centers. Similary it does for the other half. 
	Finally both the cluster centers are merged and again 
	runs kmeans to find final cluster centers for that 
	particular object descriptors.

	@param nCenters: Number of cluster centers.
	"""
	path = os.getcwd()

	#Create directory Center if it does not exists
	if not os.path.exists('Centers'):
	    os.makedirs('Centers')

	os.chdir('Desc/')

	criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 0.0001)

	for i in os.listdir(os.getcwd()):
	    Desc = open(i,"rb") #: File pointer for descriptor file
	    center = np.zeros((0,128))	#: Populator for cluster centers

	    while 1:
	        try:                    
	            des = pickle.load(Desc)		#: Read descriptor into des(numpy array)
	            print(np.shape(des))

	            #Checking the version of opencv..
	            if __version__[0] == '3':
	            	ret,label,center1=cv2.kmeans(des,int(nCenters),None,criteria,50,cv2.KMEANS_PP_CENTERS)
	            else:
	            	ret,label,center1=cv2.kmeans(des,int(nCenters),criteria,50,cv2.KMEANS_PP_CENTERS)
	            del des
	            center = np.vstack((center,center1))	#: Append cluster centers
	            print(np.shape(center))
	        except EOFError:
	            break		#: Detect End of file and break while loop
	    
	    del center1
	    des = np.float32(center)	#Convert to float, required by kmeans
	    print(np.shape(des))
	    
	    #Checking the version of opencv..
	    if __version__[0] == '3':
	    	ret,label,center1=cv2.kmeans(des,int(nCenters),None,criteria,50,cv2.KMEANS_PP_CENTERS)
	    else:
	        ret,label,center1=cv2.kmeans(des,int(nCenters),criteria,50,cv2.KMEANS_PP_CENTERS)

	    Center = open(path+"/Centers/"+i,"wb")	#: File pointer for centers file
	    pickle.dump(center,Center)  #: Save cluster centers to file
	    Center.close()
	    Desc.close()
	del path

if __name__ == '__main__':
	if(len(sys.argv)!=2):
	    print('Error. Usage: python kMeans.py numCenters')
	else:
	    kMeans(sys.argv[1])
