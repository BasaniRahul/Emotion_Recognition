
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

dataSet = pd.read_csv("/home/rahul/project/facial-expression-work/final/fer2013_final.csv")


# In[3]:

dataSet.columns


# In[4]:

#dataSet['emotion'].value_counts()


# In[5]:

emotion_0 = dataSet[dataSet['emotion']==0]
emotion_1 = dataSet[dataSet['emotion']==1]
emotion_2 = dataSet[dataSet['emotion']==2]
emotion_3 = dataSet[dataSet['emotion']==3]
emotion_4 = dataSet[dataSet['emotion']==4]
emotion_5 = dataSet[dataSet['emotion']==5]
emotion_6 = dataSet[dataSet['emotion']==6]


# In[7]:

del dataSet


# In[9]:

print("0",emotion_0.shape)
print("1",emotion_1.shape)
print("2",emotion_2.shape)
print("3",emotion_3.shape)
print("4",emotion_4.shape)
print("5",emotion_5.shape)
print("6",emotion_6.shape[0])


# In[8]:

def takeRandom(df,fraction=0.2):
    a = df.sample(frac=1).reset_index(drop=True)
    length = int(a.shape[0]*fraction)
    return a[:length], a[length+1:]


# In[16]:

emotion_0_test , emotion_0_train = takeRandom(emotion_0)
emotion_1_test , emotion_1_train = takeRandom(emotion_1)
emotion_2_test , emotion_2_train = takeRandom(emotion_2)
emotion_3_test , emotion_3_train = takeRandom(emotion_3)
emotion_4_test , emotion_4_train = takeRandom(emotion_4)
emotion_5_test , emotion_5_train = takeRandom(emotion_5)
emotion_6_test , emotion_6_train = takeRandom(emotion_6)


# In[12]:

#emotion_1_test.shape


# In[18]:

testDataSet = pd.concat([emotion_0_test,emotion_1_test,emotion_2_test,emotion_3_test,emotion_4_test,emotion_6_test])


# In[19]:

trainDataSet = pd.concat([emotion_0_train,emotion_1_train,emotion_2_train,emotion_3_train,emotion_4_train,emotion_6_train])


# In[35]:

#print(trainDataSet['emotion'].shape)
#trainDataSet['emotion'].value_counts()


# In[33]:

print(testDataSet.shape)
testDataSet['emotion'].value_counts()


# In[24]:

def shuffleDataset(dataSet):
    return dataSet.sample(frac=1).reset_index(drop=True)


# In[25]:

testDataSet = shuffleDataset(testDataSet)
trainDataSet = shuffleDataset(trainDataSet)


# In[27]:

testDataSet['Usage']="Test"


# In[30]:

testDataSet['Usage'].value_counts()


# In[31]:

testDataSet.to_csv('/home/rahul/project/facial-expression-work/final/testDataSet.csv')


# In[ ]:

trainDataSet.to_csv('/home/rahul/project/facial-expression-work/final/trainDataSet.csv')


# In[ ]:



