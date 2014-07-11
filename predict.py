import cv2
from svmutil import *
import pickle
import shutil

m = svm_load_model('model.model')
Center = open('centerFinal.p',"rb")
center = pickle.load(Center)

sift = cv2.SIFT()
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

path = os.getcwd()
if not os.path.exists('Failed'):
    os.makedirs('Failed')
os.chdir('Cross')

c = 0
crossSet = []
label = []
for i in os.listdir(os.getcwd()):
    os.chdir(path+'/Cross/'+i)
    print(i)
    for j in os.listdir(os.getcwd()):
        img = cv2.imread(j)
        print(j)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     
        kp, desc = sift.detectAndCompute(img, None)
        #len_desc = len(kp)/2
        matches = bf.match(desc, center)
        Histogram = [0]*500
        #Histogram = [ float(i) for i in Histogram ]
        for i in matches:
            Histogram[i.trainIdx]+=1
        #Histogram = [j/len_desc for j in Histogram]
        crossSet.append(Histogram)
        label.append(c)
    c+=1
    print(len(label))

print(len(label))
p_label, p_acc, p_val = svm_predict(label, crossSet, m, '-b 0')

c=0
os.chdir(path+'/Cross/')
for i in os.listdir(os.getcwd()):
    os.chdir(path+'/Cross/'+i)
    if not os.path.exists(path+'/Failed/'+i):
        os.makedirs(path+'/Failed/'+i)
    print(i)
    for j in os.listdir(os.getcwd()):
        if(p_label[c] != label[c]):
            shutil.copy(j,path+'/Failed/'+i+'/')
        c+=1
