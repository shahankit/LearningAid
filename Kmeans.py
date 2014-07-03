import numpy as np
import cv2
from matplotlib import pyplot as plt
from Pickle_Keypoints import unpickle_keypoints
import pickle

kp1 = []
des1 = []
objs1 =[]
objs = []
Keypt = open("Keypt_Desc.p","rb")  #Open File
while 1:
    try:
        objs = pickle.load(Keypt)
        print(len(objs))
        objs1.extend(objs)
    except EOFError:
        break

des = np.array(objs1)
print (np.shape(des))
del objs1
del objs
des = np.float32(des)
criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10,0.8)
ret,label,center=cv2.kmeans(des,500,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
Center = open("Centers.p","wb")
pickle.dump(center,Center)  #Saving The Keypoints and Descriptor's
Keypt.close()
print(len(center))
