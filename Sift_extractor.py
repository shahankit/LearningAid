import os
import cPickle as pickle
import numpy as np
import cv2
from Pickle_Keypoints import pickle_keypoints

path = os.getcwd()
os.chdir(path + '/'+'Data')  #Change Directory
parentdir = os.getcwd()  #Get current Directory
ptlistdir = os.listdir(parentdir)
ptlistdir_cpy = ptlistdir
sift = cv2.SIFT()  #intialize sift object
os.chdir(path)
Keypt = open("Keypt_Desc.p","wb")  #Open File
for x in range(0,len(ptlistdir)):
    temp_array=[]
    childdir = parentdir +'/'+ ptlistdir[x] 
    childdir_cpy = childdir
    os.chdir(childdir)
    cllistdir = os.listdir(childdir)
    print(len(cllistdir))
    for y in range(0,len(cllistdir)):
        childdir_cpy = childdir + '/' + cllistdir[y]
        img = cv2.imread(childdir_cpy,0)
        kp, desc = sift.detectAndCompute(img,None)  #Computing Sift Keypoints and Descriptor's
        temp = pickle_keypoints(kp,desc)
        temp_array.extend(temp)
        print(childdir_cpy)
    os.chdir(path)  #Change Directory
    pickle.dump(temp_array,Keypt)  #Saving The Keypoints and Descriptor's
    del temp_array
Keypt.close()
