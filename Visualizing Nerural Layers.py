
# coding: utf-8

# In[1]:

import csv
import numpy as np
import tensorflow as tf


pixel_depth = 255.0

def load_csv(filename,length,force=False):
    num = 0
    data = np.ndarray(shape=(length,48,48),dtype=np.float32)
    labels = np.ndarray(shape=(length,),dtype=np.float32)
    with open(filename) as csvfile:
        fer2013 = csv.reader(csvfile)
        for row in fer2013:
            if force == False:
                force = True
            else:
                d = row[1].split(" ")
                l = row[0]
                d = np.array(d,dtype=float).reshape((48,48))
                l = np.array(l,dtype=float)
                data[num,:,:] = d
                labels[num,] = l
                num = num + 1
                
    #normalized image dataset
    data = (data[0:num,:,:]-128.0)/128.0
    labels = labels[0:num,]
    print('Full dataset tensor:', data.shape)
    print('Label of dataset: ',labels.shape)
    print('Mean:', np.mean(data))
    print('Standard deviation:', np.std(data))
    return data,labels    


# In[2]:

import pandas as pd
train = pd.read_csv("testSample_v06.csv")
print(train.shape)
#del train


# In[3]:

test_dataset,test_labels = load_csv("testSample_v06.csv",train.shape[0])


# In[4]:

image_size = 48
num_labels = 7
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
    dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
#train_dataset, train_labels = reformat(train_dataset, train_labels)
#valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
#print('Training set', train_dataset.shape, train_labels.shape)
#print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[83]:


sampleDataset = test_dataset[3100:3116]
sampleDataset.shape


# In[7]:

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 32


layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
layer1_biases = tf.Variable(tf.zeros([depth]))
layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
conv = tf.nn.conv2d(sampleDataset, layer1_weights, [1, 1,1,1], padding='SAME')

hidden1 = tf.nn.relu(conv + layer1_biases)
conv2 = tf.nn.max_pool(hidden1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
conv3 = tf.nn.conv2d(conv2, layer2_weights, [1,1,1, 1], padding='SAME')

hidden2 = tf.nn.relu(conv3 + layer2_biases)
hidden3 = tf.nn.max_pool(hidden2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# In[9]:

#initilizing session
inin = tf.initialize_all_variables()
sess = tf.Session()
sess.run(inin)


# In[95]:

c= conv.eval(session=sess)


# In[99]:

c2= conv2.eval(session=sess)
c3= conv3.eval(session=sess)


# In[104]:

h1= hidden1.eval(session=sess)
h2= hidden2.eval(session=sess)
h3= hidden3.eval(session=sess)


# In[15]:


import math
import cv2


# In[114]:

#function to print conv/hidden layer
def plotNNFilter(units,l):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        a=units[i,:,:,i]*128
        a = cv2.resize(a,(60,60), cv2.INTER_LINEAR)
        cv2.imwrite("img{0}_{1}.jpg".format(l,i),a)
        #plt.show()


# In[116]:

plotNNFilter(c,"conv")
plotNNFilter(c2,"conv2")
plotNNFilter(c3,"conv3")
plotNNFilter(h1,"hid1")
plotNNFilter(h2,"hid2")
plotNNFilter(h3,"hid3")

