#!/usr/bin/env python
# coding: utf-8

# In[1]:


#in this code we process the csv generated from the big query i.e big_query.csv  and save the processed csv as processed_data.csv



import os
import pandas as pd

filepath = os.getcwd()+"/dataset/big_query.parquet"


# In[2]:


import findspark
findspark.init()

import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# In[3]:


data_frame = spark.read.parquet(filepath)


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


# In[5]:


from bs4 import BeautifulSoup
import lxml
def pre_process(x):                   #remove the code section
    #updating questions i.e removing all the html tags using parsers
    #print(x)
    soup = BeautifulSoup(x, 'lxml')
    if soup.code: soup.code.decompose()     # Remove the code section
    tag_p = soup.p
    tag_pre = soup.pre
    text = ''
    if tag_p: text = text + tag_p.get_text()
    if tag_pre: text = text + tag_pre.get_text()
    #print(tag_pre,tag_p)
    return text


# In[6]:


from textblob import TextBlob
def TextBlob_1(x):
    return TextBlob(x).polarity


# In[7]:


from pyspark.sql.functions import *
from pyspark.sql.types import *

#print(data_frame.body)
udf_myFunction = udf(pre_process, StringType())
#print((data_frame.schema.names))


#removing all the html tags from body and answers and titles and forming new columns for the same
data_frame_procc = data_frame.withColumn('processed_body',udf_myFunction(data_frame.body))

data_frame_procc_1 = data_frame_procc.withColumn('processed_title',udf_myFunction(data_frame.title))
data_frame_final = data_frame_procc_1.withColumn('processed_answers',udf_myFunction(data_frame.answers))
data_frame_final = data_frame_final.withColumn('sentiment',udf(TextBlob_1, FloatType())(data_frame_final.answers))


# In[8]:


from pyspark.sql import functions as sf

#concatenating title,body, answers into joined_data
df_new_col = data_frame_final.withColumn('joined_data', 
                    sf.concat(sf.col('title'),sf.lit(' '), sf.col('processed_body'),sf.lit(' '),sf.col('processed_answers')))


# In[9]:


# now preprocessing the joined_data and tokenizing them and normalizing the score
data_frame_tokenized = df_new_col.withColumn('joined_data',udf(preprocess_text, ArrayType(StringType()))(df_new_col.joined_data))
min_score = data_frame_tokenized.select("score").rdd.min()[0]
max_score = data_frame_tokenized.select("score").rdd.max()[0]
mean_score = data_frame_tokenized.groupBy().avg("score").take(1)[0][0]

#normalizing the score
data_frame_toknorm = data_frame_tokenized.withColumn("score",(data_frame_tokenized.score-mean_score)/(max_score-min_score))


# In[11]:


#saving the processed dataframe into a parquet file

save_filepath = os.getcwd()+"/dataset/processed_data.parquet"
data_frame_toknorm.write.format('parquet').mode("overwrite").save(save_filepath)


# In[ ]:




