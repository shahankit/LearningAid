import os
import numpy as np
import cv2
import pickle

path = os.getcwd()
center = open("centerFinal.p","rb")
Centers = pickle.load(center)
center.close()
os.chdir(path + '/'+'Train')  #Change Directory
parentdir = os.getcwd()  #Get current Directory
ptlistdir = os.listdir(parentdir)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False) #create BFMatcher Object
sift =cv2.SIFT()  #intialize sift object
Hist_nm = []
temp_array = []
label = []
for x in range(0,len(ptlistdir)):
    childdir = parentdir +'/'+ ptlistdir[x] 
    childdir_cpy = childdir
    os.chdir(childdir)
    cllistdir = os.listdir(childdir)
    print('In '+ptlistdir[x])
    for y in range(0,len(cllistdir)):
        label.append(x)
        Histogram = [0]*500
        Histogram = [ float(i) for i in Histogram ]
        childdir_cpy = childdir + '/' + cllistdir[y]
        img = cv2.imread(childdir_cpy,0)
        print(childdir_cpy)
        kp, desc = sift.detectAndCompute(img,None)  #Computing Sift Keypoints and Descriptor's
        #len_desc = len(kp)
        #len_desc = len_desc/2
        matches = bf.match(desc,Centers) #Matching Feature
        print(childdir_cpy)
        for i in matches:
            Histogram[i.trainIdx]+=1
        #Histogram = [j/len_desc for j in Histogram]
        temp_array.append(Histogram)
        del Histogram

os.chdir(path)
center.close()
Database = open("Database.p","wb")
pickle.dump(temp_array,Database)
pickle.dump(label,Database)
Database.close()
