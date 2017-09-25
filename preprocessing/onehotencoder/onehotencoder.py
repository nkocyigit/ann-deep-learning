#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Mon Sep 25 21:09:40 2017

__author__  = "nihat kocyigit"
__license__ = "GPL"
__version__ = "1.0.0"
__email__   = "nihatkoc@gmail.com"
__status__  = "Production"

#==============================================================================
# english
# In many practical Data Science activities, the data set will contain 
# categorical variables. These variables are typically stored as text values 
# which represent various traits. Some examples include color 
# (“Red”, “Yellow”, “Blue”), size (“Small”, “Medium”, “Large”) or 
# geographic designations (State or Country). Regardless of what the value 
# is used for, the challenge is determining how to use this data in the 
# analysis. Many machine learning algorithms can support categorical values 
# without further manipulation but there are many more algorithms that do not. 
# Therefore, the analyst is faced with the challenge of figuring out how to 
# turn these text attributes into numerical values for further processing.

# Label encoding has the advantage that it is straightforward but it has the 
# disadvantage that the numeric values can be “misinterpreted” by the 
# algorithms. For example, the value of 0 is obviously less than the value 
# of 4 but does that really correspond to the data set in real life?

# A common alternative approach is called one hot encoding. 
# Despite the different names, the basic 
# strategy is to convert each category value into a new column and assigns 
# a 1 or 0 (True/False) value to the column. This has the benefit of not 
# weighting a value improperly but does have the downside of adding more 
# columns to the data set.

# code uses the function from sklearn
#==============================================================================

#==============================================================================
# turkish

# Bircok pratik Data Science aktivitesinde, veri setinde kategorik degiskenler 
# yer alacaktir. Bu degişkenler genellikle cesitli ozellikleri temsil eden 
# metin degerleri olarak saklanir. Bazı ornekler arasinda renk 
# ("Kirmizi", "Sari", "Mavi"), boyut ("Kucuk", "Orta", "Buyuk") veya cografi 
# belirtimler (Eyalet veya Ulke) bulunur. Degerin ne icin kullanildigina 
# bakilmaksizin, analizde bu verilerin nasil kullanilacagini belirlemek 
# zordur. Pek cok makine ögrenme algoritmasi, kategorik degerleri baska 
# manipulasyon olmadan destekleyebilir, ancak bunu yapmayan daha bircok 
# algoritma vardir. Bu nedenle, analist, bu metin niteliklerini sonraki 
# isleme icin sayisal degerlere donusturme problemi ile karsi karsiya kalir.

# Label encoding basit olmasi avantajlidir, ancak sayisal degerlerin 
# algoritmalar tarafindan "yanlis yorumlanmasi" gibi dezavantaja sahiptir. 
# Ornegin, 0 degeri aciktir ki 4 degerinden kucuktur, ancak bu gercek 
# yasamdaki veri kumesine karsilik gelmekte midir?

# Ortak bir alternatif yaklasim, OneHotEncoding olarak adlandirilir 
# Farkli adlara ragmen, temel strateji her kategori degerini yeni bir sutuna 
# donusturmek ve sutuna 1 veya 0 (Dogru / Yanlis) degeri atamaktir. Bu, bir 
# degerin yanlis bir sekilde agirliklandirilmama avantajina sahiptir; ancak, 
# veri kumesine daha fazla sutun ekleme dezavantajina sahiptir.

# sklearn kutuphanesinden ilgili fonksiyon kullanilmaktadir
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

# d categorical value type needs d-1 extra column
# for gender feature we have 2 categories male/female so we need d = 2 field,
# he country type has 3 types Spain/France/Germany, for the country
# feature we need d = 3 extra fields, so we need to use onehotencoder
one_hot_encoder = preprocessing.OneHotEncoder(categorical_features = [1] ,sparse=True)
X = one_hot_encoder.fit_transform(X).toarray()
X = X[:, :]
X.tolist()
