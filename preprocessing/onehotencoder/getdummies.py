#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Mon Sep 25 22:55:27 2017

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

# find the path to the dataset
looking_for = 'ann-deep-learning'
current_path = os.path.dirname(os.path.realpath('__file__'))
index = current_path.find(looking_for)
root_path = current_path[:index + len(looking_for)]

# importing the dataset, specifying the seperator as ";"
dataset = pd.read_csv(root_path+'/datasets/cars.csv', sep=';')
print(dataset.shape)

# print the dataset column names
columns = dataset.columns
print(columns)

# columns for the one hot encoding using the pandas get_dummies method
# this will add the fields 'Origin_CAT', 'Origin_Europe', 'Origin_Japan',
# 'Origin_US'
cols_to_transform = [ 'Origin' ]
dataset_with_dummies = pd.get_dummies(dataset, columns = cols_to_transform )

# let's print the new dataset with the dummy fields
# 'Origin_CAT', 'Origin_Europe', 'Origin_Japan','Origin_US'
columns = dataset_with_dummies.columns
print(columns)
