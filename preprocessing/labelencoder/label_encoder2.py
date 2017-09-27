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

# importing the dataset, specifying the seperator as ";"
dataset = pd.read_csv(root_path+'/datasets/cars.csv', sep=';')
print(dataset.shape)

# get rows start from row 1 (inclusive) and columns start from 1 inclusive
X = dataset.iloc[1:, 1:].values

# we can see that our dataset include some categorical data like
# country names or gender, so we will convert this categorical data
# to encoded or numerical data
# let's transform the origin field, its column index is 7
label_encoder_X7 = preprocessing.LabelEncoder()
label_encoder_X7.fit(X[:, 7])
X[:, 7] = label_encoder_X7.transform(X[:, 7])
print(X)

# also we can do fit and transform in one line of code
# reset the X
X = dataset.iloc[1:, 1:9].values

# fit and transform for country field
label_encoder_X7 = preprocessing.LabelEncoder()
X[:, 7] = label_encoder_X7.fit_transform(X[:, 7])
print(X)
