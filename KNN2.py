# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:20:59 2020

@author: 胖大海
"""


import tensorflow as tf
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles

DATA_TYPE = 'blobs'

#value of k
k_2=150

#Number of all points
N=600

#Number of test_points
N2=100
# Number of clusters, if we choose circles, only 2 will be enough
if (DATA_TYPE == 'circle'):
    K=2
else:
    K=4


start = time.time()

#generate the points
centers = [(-2, -2), (-2, 1.5), (1.5, -2), (2, 1.5)]
if (DATA_TYPE == 'circle'):
    data, features = make_circles(n_samples=N, shuffle=True, noise= 0.01, factor=0.4)
else:
    data, features = make_blobs (n_samples=N, centers=centers, n_features = 2, cluster_std=0.8, shuffle=True, random_state=42)

fig, ax = plt.subplots()
if (DATA_TYPE == 'blobs'):
    ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1], marker = 'o', s = 250)
    ax.scatter(data.transpose()[0], data.transpose()[1], marker = 'o', s = 100, c = features, cmap=plt.cm.coolwarm )
    print("G2")
    plt.show()


points=tf.Variable(data)

assignments=tf.Variable(features)
train_assignments= tf.Variable(tf.slice(assignments.initialized_value(), [0], [N-N2]))

#取出后50个tensor作为test数据
test_points = tf.Variable(tf.slice(points.initialized_value(), [N-N2,0], [N2,2]))
train_points= tf.Variable(tf.slice(points.initialized_value(), [0,0], [N-N2,2]))

#every test point
te_point = tf.placeholder("float64", [1,2])

rep_test_point  = tf.tile(te_point, [N-N2, 1])

distance=tf.reciprocal(tf.reduce_sum(tf.square(train_points - rep_test_point),1))

#nearest = tf.argmin(distance, 0)
nearest=tf.nn.top_k(distance,k_2)

accuracy = 0.
wrong_cnt = 0

cnt= tf.Variable(tf.zeros([K], dtype=tf.float64))
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    ptest_points=sess.run(test_points)

    # loop over test data
    for i in range(N2):
       # Get nearest neighbor
        ptest_points=sess.run(tf.slice(test_points,[i,0],[1,2]))
#        print(ptest_points)
        index = sess.run(nearest[1], feed_dict={te_point: ptest_points})
        
        #classify based on k_2 points
        point_cnt=sess.run(cnt)
        for j in range(0,k_2):
            point_cnt[sess.run(assignments)[index[j]]] +=1
         
        print(point_cnt,end='')
#        Get nearest neighbor class label and compare it to its true label
        print("Test", i+1, "Prediction:", np.argmax(point_cnt), \
            "True Class:", sess.run(assignments)[N-N2+i],end='')
        # Calculate accuracy
        if np.argmax(point_cnt) == sess.run(assignments)[N-N2+i]:
            accuracy += 1./N2
            print('\n')
        else:
            wrong_cnt += 1
            print("  wrong \n")

    print("当k=",k_2,"时,",N2,"次分类中错误",wrong_cnt,"次 ","Accuracy:", accuracy)
    fig, ax = plt.subplots()
    ax.scatter(sess.run(train_points).transpose()[0], sess.run(train_points).transpose()[1], marker = 'o',c =  sess.run(train_assignments), s = 100, cmap=plt.cm.coolwarm )
    plt.show()

end = time.time()
print(("end in %.2f seconds" % (end-start)))

