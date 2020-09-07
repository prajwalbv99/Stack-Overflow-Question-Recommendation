import re
import nltk
import inflect
from nltk.corpus import stopwords
import spacy
EN = spacy.load('en_core_web_sm')

stopwords_english = stopwords.words('english')

import numpy as np
import os
from pyspark.ml.feature import Word2Vec, Word2VecModel

# Global model load
bucket_name = "cloud-stack-overflow"
s3_dir = "word2vecmodel"
local_dir = "/tmp"
title_vector = "title_vectors.parquet"
figures = "fig"

def startup():
    # Download the model
    import findspark
    findspark.init()

    import pyspark # only run after findspark.init()
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    print("Spark session started\n\n")

    aws_model_cmd = f"aws s3 cp --recursive s3://{bucket_name}/{s3_dir} {local_dir}/{s3_dir}"
    # os.system(aws_model_cmd)

    # Download the vectors
    aws_vector_cmd = f"aws s3 cp --recursive s3://{bucket_name}/{title_vector} {local_dir}/{title_vector}"
    # os.system(aws_vector_cmd)

    figures_cmd = f"aws s3 cp --recursive s3://{bucket_name}/{figures} dashboard/templates/static"
    os.system(figures_cmd)

    # Load the Model and data
    saveword2vec_path = f"{local_dir}/{s3_dir}"
    model_word2vec = Word2VecModel.load(saveword2vec_path)
    title_vectors_df = spark.read.parquet(f"{local_dir}/{title_vector}")

    return spark, model_word2vec, title_vectors_df

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

def cos_sim(d,c,a,b):
    if np.dot(a,b)==0:
        return 0
    return 0.4*d+0.1*c+float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))