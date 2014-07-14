"""
This module trains and saves the svm model. It 
reads the features and label from Database.p file.
"""

import pickle
from svmutil import *

Database = open("Database.p", "rb")		#: File pointer for Database file
features = pickle.load(Database)		#: Load features
label = pickle.load(Database)			#: Load label
prob = svm_problem(label, features)		#: Create SVM problem
param = svm_parameter('')		#: Set parameters for SVM, '' sets to default
m = svm_train(prob, param)		#: Train SVM model.
svm_save_model('model.model', m)	#: Save SVM model to file