#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:04:24 2018

@author: Grace Gibson
"""
#This file is set up for supervised text classification and will not apply unless we have a clear end-target
#including expected words or a list of focus words
#import interface to connect to postgres
hostname = 'localhost'
username = 'postgres'
password = 'aw3s0me!'
database = 'notable'

import psycopg2
print "Using psycopg2 in basicClass..."
myConnection = psycopg2.connect(host=hostname, user=username, password=password, dbname=database )

import pandas as pd
from scipy import sparse

#set up port to connect#
#s = socket.socket()
#host = 'localhost'
#port = 5432
#s.connect((host, port))
    
#s.recvfrom()
#s.close()

from query import doQuery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#from query import testModel

#define word features
def features(sentence, index):
    """ sentence: sum([word1, word2, word3...]), index: position of word"""
    return{
            'word': sentence[index],
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1, #index starts at 0
            'is_capitalized': sentence[index][0].upper() == sentence[index][0],
            'is_all_caps': sentence[index].upper() == sentence[index],
            'is_all_lower': sentence[index].lower() == sentence[index],
            'prefix-1': sentence[index][0],
            'prefix-2': sentence[index][:2],
            'prefix-3': sentence[index][:3],
            'suffix-1': sentence[index][-1:],
            'suffix-2': sentence[index][-2:],
            'suffix-3': sentence[index][-3:],
            'prev_word': ''if index == 0 else sentence[index - 1],
            'next_word': ''if index == len(sentence) - 1 else sentence[index + 1],
            'has_hyphen': '-' in sentence[index],
            'is_numeric': sentence[index].isdigit(),
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
            }
    
    
#transform data set
#get data and turn into tokens
tokens = doQuery(myConnection)
cutoff = int(.80 * len(tokens))
train_sents = tokens[:cutoff]
train_sentences = str(tokens)
test_sents = tokens[cutoff:]

#Split data into train and test
cutoff = int(.80 * len(train_sents))
train_sents = train_sents[:cutoff]
test_sents = train_sents[cutoff:]

#write tokens to text file
import os
home_dir = os.path.expanduser('~')
desktop_dir = os.path.join(home_dir, 'Desktop')
#open text file to write
count_vect = CountVectorizer()
with open(os.path.join(desktop_dir, 'textFile.txt'),'r+') as saveFile:
    for token in tokens:
        string = str(token)
        saveFile.write(string)
    X_train_counts = count_vect.fit_transform(saveFile)

print X_train_counts.shape


transformer = TfidfTransformer()
X_train_tfidf = transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape
        



#Ensure connection is closed when done
myConnection.close()
