#!/usr/bin/env python
# coding: utf-8

#importing libraries
import cv2
import scipy
import os
import scipy.misc
import time
import matplotlib.pyplot as mat
import numpy as np
import pickle
import re
from sklearn.externals import joblib
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras
from keras import applications
from PIL import Image


assault = {}
def get_me_seconds(obtained_string):
    return (int(obtained_string.split(':')[0])*60+int(obtained_string.split(':')[1]))
def let_us_club_and_flatten_it(x):
    l=np.array([])
    for i in range(0,len(x),2):
        l = np.append(l, np.arange(x[i],x[i+1]+1))
    return l.flatten()

for x in open('/Users/deepak/crime_detection/assault1csv.csv').readlines():
    temp_var = (x.replace("\n","").split(','))
    assault[temp_var[0].replace(' ',"")] = let_us_club_and_flatten_it([get_me_seconds(x) for x in temp_var[1:]])


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(data, key=alphanum_key)
class_labels = []
sample = 'assault'
for x in sorted_aphanumeric(os.listdir('/Users/deepak/crime_detection/crime_frames/assault')):
    l=int(x.split(' ')[1].split('.')[0])
    name_in_dict = x.split('.')[0]
    if int(name_in_dict[len(sample):]) <= 25:
        if l in assault[name_in_dict]:
            class_labels.append(1)
        else:
            class_labels.append(0)
    else:
        break
vectors_assault = joblib.load('crime_images_to_vectorsAssault.txt')
vectors_assault = vectors_assault[:len(class_labels)]
class_labels = np.array(class_labels).reshape((len(vectors_assault),1))

final_train_x = np.array([x/255.0 for x in vectors_assault[:2000]])
final_train_y = class_labels[:2000]
final_test_x = np.array([x/255.0 for x in vectors_assault[2000:]])
final_test_y = class_labels[2000:]

sum1 = 0
sum0 = 0
for x in np.array(class_labels).flatten():
    if x==1:
        sum1 = sum1+1
    else : 
        sum0 = sum0+1

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
x_train,y_train=sm.fit_sample(vectors_assault.reshape(len(vectors_assault),230*230*3),np.array(class_labels))
x_Train = np.array(x_train).reshape(3708,230,230,3)
y_Train = y_train.reshape(3708,1)

rng_state = np.random.get_state()
np.random.shuffle(x_Train)
np.random.set_state(rng_state)
np.random.shuffle(y_Train)

final_train_x = np.array([x/255.0 for x in x_Train[:3200]])
final_train_y = y_Train[:3200]
final_test_x = np.array([x/255.0 for x in x_Train[3200:]])
final_test_y = y_Train[3200:]

#tensorflow code for testing
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, 230, 230, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 13])
#convolution1
W_conv1 = weight_variable([3, 3, 3, 16])
b_conv1 = bias_variable([16])

h_conv1 = (conv2d(x, W_conv1, 1) + b_conv1)
batch_normalisation=tf.layers.batch_normalization(h_conv1,axis=3)
batch_normalisation_after_applying_relu=tf.nn.relu(batch_normalisation)

#maxpooling1
max_pool_1=tf.nn.max_pool(batch_normalisation_after_applying_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

#second convolutional layer

W_conv2=weight_variable([3,3,16,16])
b_conv2=bias_variable([16])

h_conv2=(conv2d(max_pool_1,W_conv2,1)+b_conv2)
batch_normalisation2=tf.layers.batch_normalization(h_conv2,axis=3)
batch_normalisation2_after_applying_relu=tf.nn.relu(batch_normalisation2)

#maxpooling 2
max_pool_2=tf.nn.max_pool(batch_normalisation2_after_applying_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

#convolution 3
W_conv3=weight_variable([3,3,16,32])
b_conv3=bias_variable([32])

h_conv3=(conv2d(max_pool_2,W_conv3,1)+b_conv3)

batch_normalisation3=tf.layers.batch_normalization(h_conv3,axis=3)
batch_normalisation3_after_applying_relu=tf.nn.relu(batch_normalisation3)

#maxpooling 3
max_pool_3=tf.nn.max_pool(batch_normalisation3_after_applying_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

#convolution 4

W_conv4=weight_variable([3,3,32,32])
b_conv4=bias_variable([32])
h_conv4=(conv2d(max_pool_3,W_conv4,1)+b_conv4)

batch_normalisation4=tf.layers.batch_normalization(h_conv4,axis=3)
batch_normalisation4_after_applying_relu=tf.nn.relu(batch_normalisation4)
keep_prob_0=tf.placeholder(tf.float32)
drop_out_addition=tf.nn.dropout(batch_normalisation4_after_applying_relu,keep_prob=keep_prob_0)
#maxpooling 4

max_pool_4=tf.nn.max_pool(drop_out_addition,ksize=[1 ,2,2,1],strides=[1,2,2,1],padding='VALID')

max_pool_flatten=tf.reshape(max_pool_4,[-1,12*12*32])
keep_prob1=tf.placeholder(tf.float32)
w_fc_1=weight_variable([4608,512])
b_fc_1=weight_variable([512])

before_final_activation=tf.nn.relu(tf.matmul(max_pool_flatten,w_fc_1)+b_fc_1)
before_final_activation_drop_out=tf.nn.dropout(before_final_activation,keep_prob=keep_prob1)
w_fc_2=weight_variable([512,13])
b_fc_2=weight_variable([13])

final_activation_y=(tf.matmul(before_final_activation,w_fc_2)+b_fc_2)
session = tf.InteractiveSession()
L2NormConst = 0.001
train_vars = tf.trainable_variables()
num_inputs = 1152
# Num of steps in each batch
num_time_steps = 30
# 100 neuron layer, play with this
num_neurons = 250
# Just one output, predicted time series
num_outputs = 1
learning_rate = 0.001
num_train_iterations = 6000
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.GRUCell(num_units=225, activation=tf.nn.relu),
    output_size=num_outputs) 
outputs, states = tf.nn.dynamic_rnn(cell, h_conv_lstm_1, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(outputs - y)) # MSE

optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
#Gradient Clipping
session.run(tf.global_variables_initializer())
epochs = 30
batch_size = 200
less=100000
loss_iteration=[]
saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)
# train over the dataset about 30 times
for epoch in range(600000):
    for i in range(510):
        optimizer.run(feed_dict={x:final_train_img[i:i+30], y_:final_class_label_vector[i:i+30],keep_prob1:0.2,keep_prob_0:0.5})
        if i % 10 == 0:
            loss_value = loss.eval(feed_dict={x:final_train_img[i:i+30], y_:final_class_label_vector[i:i+30],keep_prob1:0.2,keep_prob_0:0.5})
            print 'The number of epoch',epoch,'The loss is ',loss_value
            loss_iteration.append(loss_value)
            if loss_value<less:
                less=loss_value
                save_path = saver.save(session, "save/model_animal_disease.ckpt")

model = Sequential()

model.add(Conv2D(16, (3,3), input_shape=(230,230,3)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))         
model.add(Conv2D(16, (3,3)))                    
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))       
model.add(Conv2D(32, (3,3)))                    
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))          
model.add(Conv2D(32, (3,3)))                     
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))  
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(np.array(final_train_x),np.array(final_train_y),
          batch_size=30,
          epochs=10000,
          verbose=1,
          validation_data=(np.array(final_test_x),np.array(final_test_y)))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

json_file = open('/Users/deepak/crime_detection/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/Users/deepak/crime_detection/model.h5")
print("loaded model")

j = s.split('\n')[1:]
training_loss = []
validation_loss = []
training_acc = []
validation_acc = [] 
for x in (range(1,len(s.split('\n')[1:])-1,2)):
    training_loss.append(j[x].split('-')[2].split(':')[1])
    validation_loss.append(j[x].split('-')[4].split(':')[1])
    training_acc.append(j[x].split('-')[3].split(':')[1])
    validation_acc.append(j[x].split('-')[5].split(':')[1])
mat.plot([float(x) for x in training_loss], label="training loss")
mat.plot([float(x) for x in validation_loss], label="validation loss")
mat.legend(loc='upper right')