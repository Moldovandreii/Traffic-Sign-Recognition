# -*- coding: utf-8 -*-
"""
Created on Sun May 17 03:25:25 2020

@author: Andrei
"""


import os
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd


path = "myData" # folder with all the class folders
labelFile = 'labels.csv' # file with all names of classes
steps_per_epoch_val=2000
epochs_val=10
imageDimesions = (32,32,3)
testRatio = 0.2   # if 1000 images split will 200 for testing
validationRatio = 0.8 # if 1000 images 20% of remaining 800 will be 160 for validation
###################################################


############################### Importing of the Images
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes.....")
for x in range (0,len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end =" ")
    count +=1
print(" ")

images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
                for image in images]

images = np.array(images32)
classNo = np.array(classNo)

#print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(classNo)), len(images)))

############################### Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

############################### READ CSV FILE
data=pd.read_csv(labelFile)

############################### PREPROCESSING THE IMAGES

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)     # CONVERT TO GRAYSCALE
    img = equalize(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img


#X_train=np.array(list(map(preprocessing,X_train)))  # TO IRETATE AND PREPROCESS ALL IMAGES
#X_validation=np.array(list(map(preprocessing,X_validation)))
#X_test=np.array(list(map(preprocessing,X_test)))
#


def lrelu(x):
    return tf.maximum(0.01 * x, x)

def conv(input, num_outputs, name=None):
    return tf.contrib.layers.convolution2d(input, num_outputs, kernel_size = (5,5), stride = (1,1), 
                                           padding = "SAME", activation_fn = lrelu,
                                           normalizer_fn = tf.contrib.layers.batch_norm)

def pool(input):
    return tf.contrib.layers.max_pool2d(input, kernel_size = (2,2), 
                                        stride = (2,2), padding = "SAME")


#graph = tf.Graph()
#
## Create model in the graph.
#with graph.as_default():
#    # Placeholders for inputs and labels.
#    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
#    labels_ph = tf.placeholder(tf.int32, [None])
#
#    # Flatten input from: [None, height, width, channels]
#    # To: [None, height * width * channels] == [None, 3072]
#    images_flat = tf.contrib.layers.flatten(images_ph)
#
#    # Fully connected layer. 
#    # Generates logits of size [None, 62]
#    logits = tf.contrib.layers.fully_connected(images_flat, 62, lrelu)
#
#    # Convert logits to label indexes (int).
#    # Shape [None], which is a 1D vector of length == batch_size.
#    predicted_labels = tf.argmax(logits, 1)
#
#    # Define the loss function. 
#    # Cross-entropy is a good choice for classification.
#    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))
#
#    # Create training op.
#    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#
#    # And, finally, an initialization op to execute before training.
#    init = tf.global_variables_initializer()
    

    
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])
    
    conv1 = conv(images_ph, 8)
    pool1 = pool(conv1)
    
    conv2 = conv(pool1, 12)
    pool2 = pool(conv2)

    conv3 = conv(pool2, 16)
    pool3 = pool(conv3)
    
    flat = tf.contrib.layers.flatten(pool3)
    logits = tf.contrib.layers.fully_connected(flat, 43, lrelu)
    predicted_labels = tf.argmax(logits, 1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))
    train = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss, global_step = global_step)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
session = tf.Session(graph=graph)

# First step is always to initialize all variables. 
# We don't care about the return value, though. It's None.
_ = session.run([init])

#print("images", X_train)


for i in range(201):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: X_train, labels_ph: y_train})
    if i % 10 == 0:
         print("Loss: ", loss_value)
         

predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: X_test})[0]
# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(y_test, predicted)])
accuracy = match_count / len(y_test)
print("\nAccuracy: {:.3f}".format(accuracy))



# Close the session. This will destroy the trained model.
session.close()



































