#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


#in this we load the saved word2vec model, generate the vectors for all the titles and save it into /dataset/title_vectors.parquet
from pyspark.ml.feature import Word2Vec,Word2VecModel

saveword2vec_path = os.getcwd()+'/dataset/word2vecmodel'


# In[4]:


model_word2vec = Word2VecModel.load(saveword2vec_path)


# In[5]:


#now I am making a titles dataframe with the title and the corresponding vector
data_frame = spark.read.parquet(filepath)


# In[6]:


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


# In[8]:


from pyspark.sql.functions import *
from pyspark.sql.types import *
titles_dataframe = data_frame

titles_dataframe_tokenized = titles_dataframe.withColumn('joined_data',udf(preprocess_text, ArrayType(StringType()))(titles_dataframe.processed_title))


# In[9]:


titles_df_results = model_word2vec.transform(titles_dataframe_tokenized)


# In[11]:


#titles_df_results.take(1)


# In[13]:


save_filepath = os.getcwd()+"/dataset/title_vectors.parquet"
titles_df_results.write.mode("overwrite").format('parquet').save(save_filepath)


# In[ ]:




