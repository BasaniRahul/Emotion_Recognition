
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[13]:

def cleanDataSet(dropAdres,dataSet):
    dropList = pd.read_csv(dropAdres)
    dropList = dropList['index'].tolist()
    #print(np.sort(dropList).head())
    dataSet = pd.read_csv(dataSet)
    dataSet.columns
    dataSet.drop(dataSet.index[dropList],inplace=True)
    return dataSet


# In[15]:

cleanData = cleanDataSet('/home/rahul/project/facial-expression-work/final/to_be_rejected_kaggle_final.csv','/home/rahul/project/facial-expression-work/fer2013.csv')


# In[16]:

cleanData.to_csv("/home/rahul/project/facial-expression-work/final/fer2013_final.csv",index=False)


# In[17]:

cleanData.shape

