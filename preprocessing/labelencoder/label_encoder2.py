#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Sun Sep 24 15:53:11 2017

__author__  = "nihat kocyigit"
__license__ = "GPL"
__version__ = "1.0.0"
__email__   = "nihatkoc@gmail.com"
__status__  = "Production"

#==============================================================================
# english
# encoding categorical data in dataset
# this code piece demonstrates the label encoding on a real world dataset
# the basic idea of label encoding is given in "label_encoder.py"
#==============================================================================

#==============================================================================
# turkish
# veri setindeki kategorik verinin numerik olarak gosterimi
# bu kod parcasi kategorik verilerin veri isleme surecine girmeden once
# etiketlenmesini gercek dunyadan alinmis bir veri seti uzerinde gostermekte
# label encoding in temel calisma sekli ve fikri "label_encoder.py" dosyasinda
# anlatilmistir
#==============================================================================

# importing the libraries
import pandas as pd
import os
from sklearn import preprocessing

# find the path to the dataset
looking_for = 'ann-deep-learning'
current_path = os.path.dirname(os.path.realpath('__file__'))
index = current_path.find(looking_for)
root_path = current_path[:index + len(looking_for)]

# importing the dataset
dataset = pd.read_csv(root_path+'/datasets/Churn_Modelling.csv')

# get all rows and all columns 3 inclusive and 13 exclusive
# column 13 is the result case or can call it y
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13:].values

# we can see that our dataset include some categorical data like
# country names or gender, so we will convert this categorical data
# to encoded or numerical data
# first let's transform the country field, its column index is 1
label_encoder_X1 = preprocessing.LabelEncoder()
label_encoder_X1.fit(X[:, 1])
X[:, 1] = label_encoder_X1.transform(X[:, 1])

# then let's transform the gender field, its column index is 2
label_encoder_X2 = preprocessing.LabelEncoder()
label_encoder_X2.fit(X[:, 2])
X[:, 2] = label_encoder_X2.transform(X[:, 2])

# also we can do fit and transform in one line of code
# reset the X
X = dataset.iloc[:, 3:13].values

# fit and transform for country field
label_encoder_X1 = preprocessing.LabelEncoder()
X[:, 1] = label_encoder_X1.fit_transform(X[:, 1])

# fit and transform for gender field
label_encoder_X2 = preprocessing.LabelEncoder()
X[:, 2] = label_encoder_X2.fit_transform(X[:, 2])
