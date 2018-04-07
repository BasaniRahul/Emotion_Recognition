
# coding: utf-8

# In[5]:

from pyspark import SparkConf,SparkContext

#creating spark context
con = SparkConf().setAppName("loading emotion csv").setMaster("local[4]")
sc = SparkContext(conf = con)
                             
#initializing start time
start_time = time.time()
                             
#reading emotion data csv file
def get_emotion_data(filename):
    rdd = sc.textFile(filename)
    emotions = rdd.map(lambda line: line.split(',')[0]).collect()
    pixels = rdd.map(lambda line: line.split(',')[1]).collect()
    data = pd.DataFrame({"emotion":emotions[1:],"pixels":pixels[1:]})
    return data

#resizing emotion data
def data_resize(filename):
    data = get_emotion_data("filePath")                         
    emotions , data_pixels = data['emotion'] , data['pixels']
    total_length = len(emotions)
    data_pixel_df = []
    emotion_list = []
    image_list = []
    for i in range(total_length):       
    
        for line in data_pixels[i].split(" "):
            data_pixel_df.append(float(line))
            
        
        data_pixel_df_array = np.array(data_pixel_df)
               
        data_pixel_df_array = data_pixel_df_array.reshape(48,48)
        resize = cv2.resize(data_pixel_df_array,(28,28),interpolation = cv2.INTER_AREA)
        
        #ab is resized image
        ab = resize.reshape(784,)
        resize_image_in_str = " ".join(str(int(a)) for a in ab)
        emotion_list.append(emotions[i])
        image_list.append(resize_image_in_str)
        data_pixel_df=[]
    resized_dataframe = pd.DataFrame({
                        'emotion': emotion_list,
                        'pixels' : image_list
                        })
    return resized_dataframe

resized_dataset = data_resize("fer2013_final.csv")



# In[ ]:

dataSet = resized_dataset

#separating dataframe as per emotions 
emotion_0 = dataSet[dataSet['emotion']==0]
emotion_1 = dataSet[dataSet['emotion']==1]
emotion_2 = dataSet[dataSet['emotion']==2]
emotion_3 = dataSet[dataSet['emotion']==3]
emotion_4 = dataSet[dataSet['emotion']==4]
emotion_5 = dataSet[dataSet['emotion']==5]
emotion_6 = dataSet[dataSet['emotion']==6]

del dataSet #to free memory

#to separate train and test dataset
def separate_train_and_test(df,fraction=0.2):
    
    #to shuffle dataset
    a = df.sample(frac=1).reset_index(drop=True)
    length = int(a.shape[0]*fraction)
    return a[:length], a[length+1:]

#separated dataset as per emotions
emotion_0_test , emotion_0_train = takeRandom(emotion_0)
emotion_1_test , emotion_1_train = takeRandom(emotion_1)
emotion_2_test , emotion_2_train = takeRandom(emotion_2)
emotion_3_test , emotion_3_train = takeRandom(emotion_3)
emotion_4_test , emotion_4_train = takeRandom(emotion_4)
emotion_5_test , emotion_5_train = takeRandom(emotion_5)
emotion_6_test , emotion_6_train = takeRandom(emotion_6)

#test dataset
testDataSet = pd.concat([emotion_0_test,emotion_1_test,emotion_2_test,emotion_3_test,emotion_4_test,emotion_5_test,emotion_6_test])

#train dataset
trainDataSet = pd.concat([emotion_0_train,emotion_1_train,emotion_2_train,emotion_3_train,emotion_4_train,emotion_5_train,emotion_6_train])

#writing datasets to csv
testDataSet.to_csv('testDataset.csv',index=False)
trainDataSet.to_csv('trainDataset.csv',index=False)


# In[ ]:

#data augmentation by mirroring and rotating
def data_augment(filename):
    
    data = pd.read_csv(filename)
    emotions , data_pixels = data['emotion'] , data['pixels']
    total_length = len(emotions)
    data_pixel_df = np.array([])
    emotion_list = np.array([])
    image_list = np.array([])
    for i in range(total_length):
        emotion_list = np.append(emotions[i])
         
        for x in data_pixels[i].split(" "):
            data_pixel_df = np.append(data_pixel_df,x)
            
        
        data_pixel_df_array = np.array(data_pixel_df,dtype = 'uint8')
        original_image = " ".join(str(a) for a in data_pixel_df_array)
        image_list = np.append(image_list,original_image)
        
        #flip image
        data_pixel_df_array = data_pixel_df_array.reshape(48,48)
        flipped_image = cv2.flip(data_pixel_df_array,1)
        
        #rotate image
        image_center = tuple(np.array(data_pixel_df_array.shape)/2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center,10,1.0)
        rotated_image = cv2.warpAffine(data_pixel_df_array,rotation_matrix,data_pixel_df_array.shape)
    
        #ab=flipped_image
        ab = flipped_image.reshape(2304,)
        flipped_image_in_str = " ".join(str(a) for a in ab)
        emotion_list = np.append(emotion_list,flipped_image)
        image_list = np.append(image_list,rotated_image)
        
        #cd=rotated_image
        cd = rotated_image.reshape(2304,)
        rotated_image_in_str = " ".join(str(c) for c in cd)
        emotion_list = np.append(emotion_list,rotated_image)
        image_list = np.append(image_list,rotated_image)
        data_pixel_df=np.array([])
    augmented_dataframe = pd.DataFrame({
                        'emotion': emotion_list,
                        'pixels' : image_list
                        })
    return augmented_dataframe

augmented_dataset = data_augment("trainDataset.csv")



# In[ ]:

dataSet =  augmented_dataset

#separating augmented_dataset as per emotions
emotion_0 = dataSet[dataSet['emotion']==0]
emotion_1 = dataSet[dataSet['emotion']==1]
emotion_2 = dataSet[dataSet['emotion']==2]
emotion_3 = dataSet[dataSet['emotion']==3]
emotion_4 = dataSet[dataSet['emotion']==4]
emotion_5 = dataSet[dataSet['emotion']==5]
emotion_6 = dataSet[dataSet['emotion']==6]

del dataset #to free memory

#to balance out less amount of data in emotion_1 : Disgust
emotion_1 = pd.concat([emotion_1]*4,ignore_index=True)

#shuffling emotion_1
emotion_1=emotion_1.sample(frac=1).reset_index(drop=True)


#separating train and valid datasets as per emotions
emotion_0_valid , emotion_0_train = takeRandom(emotion_0, fraction=0.2)
emotion_1_valid , emotion_1_train = takeRandom(emotion_1, fraction=0.2)
emotion_2_valid , emotion_2_train = takeRandom(emotion_2, fraction=0.2)
emotion_3_valid , emotion_3_train = takeRandom(emotion_3, fraction=0.2)
emotion_4_valid , emotion_4_train = takeRandom(emotion_4, fraction=0.2)
emotion_5_valid , emotion_5_train = takeRandom(emotion_5, fraction=0.2)
emotion_6_valid , emotion_6_train = takeRandom(emotion_6, fraction=0.2)

#concating all the emotions into train and valid datasets
validDataset = pd.concat([emotion_0_valid,emotion_1_valid,emotion_2_valid,emotion_3_valid,emotion_4_valid,emotion_5_valid,emotion_6_valid])
trainDataset = pd.concat([emotion_0_train,emotion_1_train,emotion_2_train,emotion_3_train,emotion_4_train,emotion_5_train,emotion_6_train])

#shuffling train and valid datasets
trainDataset = trainDataset.sample(frac=1).reset_index(drop=True)
validDataset = validDataset.sample(frac=1).reset_index(drop=True)


validDataset.to_csv("validDataset.csv",index=False)
trainDataset.to_csv("trainDataSet.csv",index=False)


# In[ ]:

import csv
import numpy as np


pixel_depth = 255.0

def load_csv(filename,length,force=False):
    num = 0
    data = np.ndarray(shape=(length,28,28),dtype=np.float32)
    labels = np.ndarray(shape=(length,),dtype=np.float32)
    with open(filename) as csvfile:
        fer2013 = csv.reader(csvfile)
        for row in fer2013:
            if force == False:
                force = True
            else:
                d = row[1].split(" ")
                l = row[0]
                d = np.array(d,dtype=float).reshape((28,28))
                l = np.array(l,dtype=float)
                data[num,:,:] = d
                labels[num,]= l
                num = num + 1
                
    #normalized image dataset
    data = data[0:num,:,:]/255.0
    labels = labels[0:num,]
    print('Full dataset tensor:', data.shape)
    print('Label of dataset: ',labels.shape)
    print('Mean:', np.mean(data))
    print('Standard deviation:', np.std(data))
    return data,labels    



# In[15]:


import pandas as pd
train = pd.read_csv("trainDataset.csv")
valid = pd.read_csv("trainDataset.csv")
test = pd.read_csv("trainDataset.csv")
length_train = train.shape[0]
length_valid = valid.shape[0]
length_test = test.shape[0]



# In[6]:

#loading train, valid and test dataset 
train_dataset,train_labels = load_csv("trainDataset.csv",length_train)


# In[16]:

test_dataset,test_labels = load_csv("testDataset.csv",length_test)


# In[11]:

valid_dataset,valid_labels = load_csv("validDataset.csv",length_valid)


# In[17]:


image_size = 28
num_labels = 7
num_channels = 1 # grayscale


#reformatting input data to make then tensorflow ready
def reformat(dataset, labels):
    dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[18]:

#calculating accuracy of train, validation and test datasets
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# In[19]:

import tensorflow as tf


# In[20]:

#hyperparameters
batch_size = 32
patch_size = 5
depth = 16
num_hidden = 64

#defining tensorflow graph
graph = tf.Graph()

with graph.as_default():

  # Input data : train dataset, validation dataset and test dataset
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
  # Variables : Weights and biases of diffent layers in CNN
    layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
    def model(data):
        #layer1
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        pool = tf.nn.max_pool(hidden, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
        #layer2
        conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        pool = tf.nn.max_pool(hidden, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        
        #fully-connected layer
        fc = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        
        #output layer
        return tf.matmul(fc, layer4_weights) + layer4_biases
  
  # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


# In[44]:

#no. of iterations
num_steps = 16001

#Session definition
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    s = []
    mini_acc = []
    val_acc = []
    losses = []
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
         [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 2000 == 0):
            min_a = accuracy(predictions, batch_labels)
            val_a = accuracy(valid_prediction.eval(), valid_labels)
            s.append(step)
            mini_acc.append(min_a)
            val_acc.append(val_a)
            losses.append(l)
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % min_a)
            print('Validation accuracy: %.1f%%' % val_a)
    x = test_prediction.eval()
    print('Test accuracy: %.1f%%' % accuracy(x, test_labels))


# In[46]:

#maximum probability value
a = np.argmax(x,1)

#actual test labels
b = np.argmax(test_labels,1)


# In[47]:

#creating dataframe of predicted and actual labels
a_pd = pd.DataFrame({"a":a})
a_pd['b'] = b


# In[48]:

#summing predicted labels
_0=np.sum(a_pd['a']==0)
_1=np.sum(a_pd['a']==1)
_2=np.sum(a_pd['a']==2)
_3=np.sum(a_pd['a']==3)
_4=np.sum(a_pd['a']==4)
_5=np.sum(a_pd['a']==5)
_6=np.sum(a_pd['a']==6)


# In[49]:

print(_0,_1,_2,_3,_4,_5,_6)


# In[50]:

#summing actual labels
b_0=np.sum(a_pd['b']==0)
b_1=np.sum(a_pd['b']==1)
b_2=np.sum(a_pd['b']==2)
b_3=np.sum(a_pd['b']==3)
b_4=np.sum(a_pd['b']==4)
b_5=np.sum(a_pd['b']==5)
b_6=np.sum(a_pd['b']==6)


# In[52]:

print(b_0,b_1,b_2,b_3,b_4,b_5,b_6)


# In[53]:

#list in the form [True,False,False,.....] for predicted values
ind_0a =(a_pd['a']==0)
ind_1a =(a_pd['a']==1)
ind_2a =(a_pd['a']==2)
ind_3a =(a_pd['a']==3)
ind_4a =(a_pd['a']==4)
ind_5a =(a_pd['a']==5)
ind_6a =(a_pd['a']==6)


# In[54]:

#list in the form [True,False,False,.....] for actual values
ind_0b =(a_pd['b']==0)
ind_1b =(a_pd['b']==1)
ind_2b =(a_pd['b']==2)
ind_3b =(a_pd['b']==3)
ind_4b =(a_pd['b']==4)
ind_5b =(a_pd['b']==5)
ind_6b =(a_pd['b']==6)


# In[55]:

#for calculating true positives 
def accu(a,b):
    sum = 0
    for i in range(len(a)):
        if b[i]:
            if a[i]==b[i]:
                sum+=1
    return sum


# In[56]:

acc_0 = accu(ind_0a,ind_0b)
acc_1 = accu(ind_1a,ind_1b)
acc_2 = accu(ind_2a,ind_2b)
acc_3 = accu(ind_3a,ind_3b)
acc_4 = accu(ind_4a,ind_4b)
acc_5 = accu(ind_5a,ind_5b)
acc_6 = accu(ind_6a,ind_6b)


# In[57]:

print("categories          (0=An, 1=Dis, 2=Fe, 3=Ha, 4=Sa, 5=Sup, 6=Neutral).")
print(" actual labels count  ",b_0,b_1,b_2,b_3,b_4,b_5,b_6)
print("predicted labels count",_0,_1,_2,_3,_4,_5,_6)
print("true positives        ",acc_0,acc_1,acc_2,acc_3,acc_4,acc_5,acc_6)

#no. of classes
n_groups = 7

index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8

#for plotting accuracy graphs
rects1 = plt.bar(index,actual,bar_width,alpha = opacity, color = 'b', label='Actual')
rects1 = plt.bar(index+bar_width,predicted,bar_width,alpha = opacity, color = 'g', label='Predicted')
rects1 = plt.bar(index+2*bar_width,correctPredicted,bar_width,alpha = opacity, color = 'y', label='Accuracy')
plt.xlabel('Emotions')
plt.ylabel('Values')
plt.title('Predicted vs Actual')
plt.xticks((index+bar_width),(0,1,2,3,4,5,6))
plt.legend()

plt.tight_layout()
plt.show()

plt.plot(s, mini_acc)
plt.plot(s, valid_acc)
plt.legend(['mini_acc','val_acc'])
plt.title('Minibatch and Validation accuracy VS Steps')
plt.show()

plt.plot(s,losses)
plt.legend(['losses'])
plt.title('Loss VS Steps')
plt.show()

