import os
import cPickle as pickle
import numpy as np
import cv2

path = os.getcwd()
if not os.path.exists('Desc'):
    os.makedirs('Desc')
os.chdir(path + '/'+'Train')  #Change Directory
parentdir = os.getcwd()  #Get current Directory
ptlistdir = os.listdir(parentdir)

sift = cv2.SIFT()  #intialize sift object
os.chdir(path)
for x in range(0,len(ptlistdir)):
    temp_array=[]
    childdir = parentdir +'/'+ ptlistdir[x] 
    childdir_cpy = childdir
    os.chdir(childdir)
    cllistdir = os.listdir(childdir)
    Keypt = open(path+'/Desc/Desc'+str(x+1)+'.p',"wb")
    num = len(cllistdir)/2
    print(len(cllistdir))
    i = 1
    for y in range(0,len(cllistdir)):
        childdir_cpy = childdir + '/' + cllistdir[y]
        print(childdir_cpy)
        img = cv2.imread(childdir_cpy)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray,None)  #Computing Sift Keypoints and Descriptor's
        temp_array.extend(desc)
        if(i==1 and y>num):	
        	pickle.dump(np.array(temp_array),Keypt)
        	temp_array = []
        	i = 0
    pickle.dump(np.array(temp_array),Keypt)  #Saving The Keypoints and Descriptor's
    Keypt.close()
    del temp_array
