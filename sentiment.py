# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:46:30 2019

@author: hp
"""

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

train_data = pd.read_csv('train_.csv')
test_data = pd.read_csv('test_.csv')
data = train_data.append(test_data, ignore_index=True)
data['tweet'] = data['tweet'].str.replace("[^a-zA-Z#]", " ")

#removing the stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

data['tweet'] = data['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

tokenized = data['tweet'].apply(lambda x: x.split())

from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized = tokenized.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

for i in range(len(tokenized)):
    tokenized[i] = ' '.join(tokenized[i])

data['tweet'] = tokenized

def hastags_collect(x):
    
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)",i)
        hashtags.append(ht)
        
    return hashtags

ht_positive = hastags_collect(data['tweet'][data['label']==0])

ht_negative = hastags_collect(data['tweet'][data['label']==1])

#
#from sklearn.feature_extraction.text import CountVectorizer
#bow_vec = CountVectorizer()
#bow = bow_vec.fit_transform(data['tweet'])
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(data['tweet'])

train_tfidf = tfidf[:7920,:]
test_tfidf = tfidf[7920:,:]

#from sklearn.model_selection import train_test_split
#xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, train_data['label'], random_state=42, test_size=0.3)
#
#xtrain_tfidf = train_tfidf[ytrain.index]
y = train_data['label']


#make a machine learning model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

reg = LogisticRegression()
reg.fit(train_tfidf, y)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
dense_ = train_tfidf.toarray()
classifier.fit(dense_, y)

from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_svm.fit(train_tfidf, y)

from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier()
random.fit(train_tfidf,y)

from sklearn.naive_bayes import MultinomialNB
classifier_multi = MultinomialNB()
classifier_multi.fit(train_tfidf, y)

from sklearn.ensemble import GradientBoostingClassifier
classifier_multi = GradientBoostingClassifier()
classifier_multi.fit(train_tfidf,y)
#from sklearn.feature_extraction.text import CountVectorizer
#bow_vec = CountVectorizer()
#bow_test = bow_vec.fit_transform(test_data['tweet'])
#from sklearn.feature_extraction.text import TfidfVectorizer
#tfidf_vectorizer_test = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
## TF-IDF feature matrix
#tfidf_test = tfidf_vectorizer_test.fit_transform(test_data['tweet'])
# 
prediction = reg.predict(test_tfidf)
prediction = reg.predict_proba(test_tfidf) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

pred_naive = classifier.predict(dense_)

pred_svm = classifier_svm.predict(test_tfidf)

pred_rf = random.predict(test_tfidf)

pred_multinomial = classifier_multi.predict(test_tfidf)
#prediction_int_multi = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
#prediction_int_multi = prediction_int.astype(np.int)

pred_GB = gb.predict(test_tfidf)

pred_multinomial_GB = classifier_multi.predict(test_tfidf)