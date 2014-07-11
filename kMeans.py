import numpy as np
import cv2
import pickle
import os

path = os.getcwd()
if not os.path.exists('Centers'):
        os.makedirs('Centers')
os.chdir('Desc/')
c = 1
criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10,0.8)
for i in os.listdir(os.getcwd()):
        Desc = open(i,"rb")  #Open File
        center = np.zeros((0,128))
        while 1:
                try:                    
                        des = pickle.load(Desc)
                        print(np.shape(des))
                        ret,label,center1=cv2.kmeans(des,500,None,criteria,50,cv2.KMEANS_PP_CENTERS)
                        del des
                        center = np.vstack((center,center1))
                        print(np.shape(center))
                except EOFError:
                        break
        des = np.float32(center)
        print(np.shape(des))
        ret,label,center2=cv2.kmeans(des,500,None,criteria,50,cv2.KMEANS_PP_CENTERS)

        Center = open(path+"/Centers/Center"+i[-3:-2]+".p","wb")
        pickle.dump(center,Center)  #Saving The Keypoints and Descriptor's
        Center.close()
        c+=1
del path
