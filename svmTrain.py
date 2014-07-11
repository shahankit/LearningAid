import pickle
import os
from svmutil import *

Dataset = open("Database.p", "rb")
features = pickle.load(Dataset)
label = pickle.load(Dataset)
prob = svm_problem(label, features)
param = svm_parameter('')
param.set_to_default_values()
m = svm_train(prob, param)
svm_save_model('model.model', m)
