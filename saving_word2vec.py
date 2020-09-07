#!/usr/bin/env python
# coding: utf-8

# In[1]:


#in this notebook, we create a word2vec model and save it into disk

#loading into file from memory
import os
import pandas as pd

filepath = os.getcwd()+"/dataset/processed_data.parquet"


# In[2]:


import findspark
findspark.init()

import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# In[3]:


data_frame = spark.read.parquet(filepath)


# In[5]:


#in this notebook, we create a word2vec model and save it into disk

from pyspark.ml.feature import Word2Vec,Word2VecModel

#creating 
word2vec = Word2Vec(vectorSize=200, seed=42, inputCol="joined_data", outputCol="features")


# In[6]:


#fitting the model with the data present
model_word2vec = word2vec.fit(data_frame)


# In[8]:


#saving the word2vec model

save_path = os.getcwd()+'/dataset/word2vecmodel'
model_word2vec.write().overwrite().save(save_path)


# In[ ]:




