#!/usr/bin/env python
# coding: utf-8

# In[1]:


#in this file, we take question as a input, generate all the requireed vectors and output a recommendation

import os
import pandas as pd
import sys

filepath = os.getcwd()+"/dataset/title_vectors.parquet"


# In[2]:


import findspark
findspark.init()

import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# In[3]:


title_vectors_df = spark.read.parquet(filepath)


# In[4]:


import re
import nltk
import inflect
from nltk.corpus import stopwords
import spacy
EN = spacy.load('en_core_web_sm')

stopwords_english = stopwords.words('english')
def tokenize_text(text):
    "Apply tokenization using spacy to docstrings."
    tokens = EN.tokenizer(text)
    return [token.text.lower() for token in tokens if not token.is_space]

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords_english:
            new_words.append(word)
    return new_words

def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

def tokenize_code(text):
    "A very basic procedure for tokenizing code strings."
    return RegexpTokenizer(r'\w+').tokenize(text)

def preprocess_text(text):
    return (' '.join(normalize(tokenize_text(text))).split(' '))


# In[6]:


####### input question
input_question = sys.argv[0]


# In[7]:


question_dataframe = spark.createDataFrame([
    (input_question, )
], ["question"])


# In[8]:


#making a joined data column with all the tokens

from pyspark.sql.functions import *
from pyspark.sql.types import *

question_tokenized_df = question_dataframe.withColumn('joined_data',udf(preprocess_text, ArrayType(StringType()))(question_dataframe.question))


# In[9]:


#print(question_tokenized_df.take(1))


# In[10]:


###now we have to generate the vectors for this given question
from pyspark.ml.feature import Word2Vec,Word2VecModel

saveword2vec_path = os.getcwd()+'/dataset/word2vecmodel'


# In[11]:


model_word2vec = Word2VecModel.load(saveword2vec_path)


# In[12]:


question_with_vector_df = model_word2vec.transform(question_tokenized_df)


# In[13]:


#taking only the dense vector
question_dense_vec = question_with_vector_df.first()["features"]


# In[14]:


#Now that we have everything in place, we just need to calculate the similarity score
import numpy as np
def cos_sim(d,c,a,b):
    if np.dot(a,b)==0:
        return 0
    return 0.4*d+0.1*c+float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# In[15]:


df_cos_sim = title_vectors_df.withColumn("similarity_score", udf(cos_sim, FloatType())(col("sentiment"),col("score"),col("features"), array([lit(v) for v in question_dense_vec])))


# In[19]:


#min_score = df_cos_sim.select("similarity_score").rdd.min()[0]
#max_score = df_cos_sim.select("similarity_score").rdd.max()[0]
#mean_score = df_cos_sim.groupBy().avg("similarity_score").take(1)[0][0]


# In[16]:


#df_cos_sim = df_cos_sim.withColumn("similarity_score",(df_cos_sim.score-mean_score)/(max_score-min_score))


# In[18]:


print(df_cos_sim.orderBy('similarity_score',ascending= False).take(2))


# In[ ]:




