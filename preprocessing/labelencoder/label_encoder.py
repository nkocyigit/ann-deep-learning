#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Sun Sep 24 14:10:35 2017

__author__  = "nihat kocyigit"
__license__ = "GPL"
__version__ = "1.0.0"
__email__   = "nihatkoc@gmail.com"
__status__  = "Production"

#==============================================================================
# english
# encoding categorical data in dataset
# this code piece demonstrates the data preprocessing
# for the labeled data, eg. gender "male","female"
# can be encoded into 0 and 1. That is necessary because
# you can not feed that kinf of data into processing phase.
# Mostly that kind of features are categorical data.
# Another example is for countries, "Turkey","USA",
# "Germany","France" can be encoded as 0,1,2 and 3.
# code uses the function from sklearn
#==============================================================================

#==============================================================================
# turkish
# veri setindeki kategorik verinin numerik olarak gosterimi
# bu kod parcasi kategorik verilerin veri isleme surecine girmeden once
# etiketlenmis bir veri ile degistirilmesini gosteren bir ornektir, ornegin
# cinsiyet "kadin","erkek" degerlerinden birisini alabilir bu yontem ile bu
# kategorik veriyi 0,1 seklinde bir veriye donusturebiliriz ya da ulkelerden
# olusan bir veri alanini Turkiye,USA,Almanya,Fransa 0,1,2,3 seklinde 
# degistirebiliriz, kategorik verinin bu sekilde degistirilmesi sonraki veri
# isleme adimlarina verinin problemsiz olarak verilebilmesi icin gereklidir
# sklearn kutuphanesinden ilgili fonksiyon kullanilmaktadir
#==============================================================================

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

countries = ["Turkey","Germany","France","Italy"]

label_encoder.fit(countries)

label_encoder.classes_

test_data = ["Turkey", "Turkey","Italy","France","Germany","Turkey"]

label_encoder.transform(test_data)
