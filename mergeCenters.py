import numpy as np
import cv2
import os
import pickle

path = os.getcwd()
os.chdir('Centers/')
des = np.zeros((0,128))
for i in os.listdir(os.getcwd()):
    Center = open(i,"rb")
    des = np.vstack((des, pickle.load(Center)))

des = np.float32(des)
criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10,0.8)
ret,label,center=cv2.kmeans(des,500,None,criteria,50,cv2.KMEANS_PP_CENTERS)

Center = open(path+'/centerFinal.p',"wb")
pickle.dump(center, Center)
Center.close()
