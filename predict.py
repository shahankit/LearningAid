"""
This module tests a given set of images in ./Test/ folder 
against the trained SVM model. It first creates a feature 
and true label vector for the images and then predicts the 
calculated label using the SVM model.

Note: The order and number of image folder in ./Train/ folder 
and ./Test/ must be same.
"""
import cv2
import os
from svmutil import *
import pickle
import shutil

histSize = 500

def featuresAndLabels():
	"""
	This function extracts features and labels for each image 
	in ./Test/ directory. It first extracts the SIFT features 
	and then maps them to cluster centers using BFMatcher.

	@return: features: List of extracted features
	@return: label: Vector containing true labels
	"""

	Center = open('centerFinal.p',"rb") #: File pointer for centers file
	center = pickle.load(Center)	#: Load centers

	sift = cv2.SIFT()	# Initalize SIFT object
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)	# Create BFMatcher object

	path = os.getcwd()

	if not os.path.exists('Failed'):
		os.makedirs('Failed')		#: Create new directory ./Failed/
	else:
		shutil.rmtree('Failed')		#: Delete already existing directory ./Failed/
		os.makedirs('Failed')		#: Create new directory ./Failed/
	os.chdir('Test')

	c = 0		#: Maintains count for labels
	testSet = []	#: Populator for test features
	label = []		#: Populator for true labels

	parentList = os.listdir(os.getcwd())
	parentList.sort()

	for i in parentList:
		os.chdir(path+'/Test/'+i)
		print(i)
		childList = os.listdir(os.getcwd())
		childList.sort()

		for j in childList:
			print(j)
			img = cv2.imread(j)
			gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     
			kp, desc = sift.detectAndCompute(gray, None)
			matches = bf.match(desc, center)
			Histogram = [0]*int(histSize)
			for i in matches:
				Histogram[i.trainIdx]+=1
			testSet.append(Histogram)
			label.append(c)
		c+=1
	
	return testSet, label

def predict():
	"""
	This function predicts the test set against the SVM 
	model. It also copies those images which are not 
	predicted correctly to ./Failed/ directory.
	"""
	path = os.getcwd()
	m = svm_load_model('model.model')   #: Load SVM model
	
	testSet, label = featuresAndLabels()

	p_label,p_acc, p_val = svm_predict(label,testSet,m,'-b 0')	#: Predict testSet against m


	#Copy wrongly recognized images to ./Failed directory.
	c=0
	os.chdir(path+'/Test/')
	parentList = os.listdir(os.getcwd())
	parentList.sort()

	for i in parentList:
		os.chdir(path+'/Test/'+i)
		os.makedirs(path+'/Failed/'+i)
		print(i)
		childList = os.listdir(os.getcwd())
		childList.sort()

		for j in childList:
			if(p_label[c] != label[c]):		#Check predicted label and true label
				shutil.copy(j,path+'/Failed/'+i+'/')
			c+=1

if __name__ == '__main__':
	if(len(sys.argv)!=2):
		print('Usage : python predict.py histogram_size')
		print('Hint : Size of histogram is equal to number of clusers')
	else:
		histSize = sys.argv[1]
		predict()
