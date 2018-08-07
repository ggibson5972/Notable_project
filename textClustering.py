#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:19:55 2018

@author: Grace Gibson
"""

#import necessary packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import random
import math
from scipy.spatial import distance

#write tokens to text file
import os
home_dir = os.path.expanduser('~')
desktop_dir = os.path.join(home_dir, 'Desktop')

vectorizer = TfidfVectorizer()
with open(os.path.join(desktop_dir, 'textFile.txt'),'r+') as saveFile:
    #Singleton array cannot be considered a valid collection
    data = saveFile.read().split(']')
    random.shuffle(data)
    cutoff = int(.8*len(data))
    train_data = data[:cutoff]
    test_data = data[cutoff:]
    vec = vectorizer.fit(train_data)
    vectorized = vec.transform(train_data)
    
#fit model around vectorized data
true_k = 9
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, tol = 0.0001, n_init = 1)
model.fit(vectorized)

lst2=vec.transform(test_data)  # transform list2 using vec
#Determine most common tokens per cluster
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])

#Bayesian Information Criterion(BIC)
#BIC(C|x) = L(x|C) - (p/2) * log(n)
def computeBIC(model, vectorized):
    centers = model.cluster_centers_
    labels = model.labels_
    m = model.n_clusters
    n = np.bincount(labels)
    N, d = vectorized.shape
    
    #Cluster variance
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(vectorized[np.where(labels == i)], [centers[0][i]], 'euclidian')**2) for i in range(m)])
    
    #L(x|C)
    constant = 0.5 * m * np.log(N) * (d + 1)
    
    #goal is to maximize because sign is inversed from regular definition
    BIC = np.sum([n[i] * np.log(n[i]) - n[i] * np.log(N) - ((n[i] * d)/2)*np.log(2*np.pi*cl_var) - ((n[i] - 1) * (d/2)) for i in range(m)]) - constant
    return BIC
    
p = 4
n = 224131
bic = p/2 * math.log(224131)
print "BIC: ", bic

#inertia = sum of squared distance sum(x - avg(X))^2
print "Model Inertia: ", model.inertia_
print("\n")
print("Prediction")

Y = vec.transform(["Called Kelly to schedule next visit"])
prediction = model.predict(Y)
print "Cluster ", prediction

Y = vec.transform(["Tried out Subculture but preferred Chocolate Lover"])
prediction = model.predict(Y)
print "Cluster ", prediction

Y = vec.transform(["Invited to presale event"])
prediction = model.predict(Y)
print "Cluster ", prediction

