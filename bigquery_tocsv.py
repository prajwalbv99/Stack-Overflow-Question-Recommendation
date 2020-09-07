#!/usr/bin/env python
# coding: utf-8

# In[2]:


import bq_helper
from bq_helper import BigQueryHelper
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="Cloud Final-1f49f4274b69.json"
bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
QUERY_1  = "SELECT ID,TITLE FROM `bigquery-public-data.stackoverflow.posts_questions` WHERE ID = 57804"
QUERY = "SELECT q.id, q.title, q.body, q.tags, a.body as answers, a.score FROM `bigquery-public-data.stackoverflow.posts_questions` AS q INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a ON q.id = a.parent_id WHERE q.tags LIKE '%python%' LIMIT 10000"

df = bq_assistant.query_to_pandas(QUERY)


# In[3]:


import findspark
findspark.init()

import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# In[5]:


file_save_path = os.getcwd()+"/dataset/big_query.parquet"
data_frame = spark.createDataFrame(df)
data_frame.write.format('parquet').mode("overwrite").save(file_save_path)


# In[4]:


print(type(df))


# In[ ]:




