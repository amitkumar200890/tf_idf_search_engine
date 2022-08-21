# Databricks notebook source
pip install nltk

# COMMAND ----------

import re
import math
import nltk
import string 
import numpy as np
import pandas as pd

from collections import Counter

from numpy import dot
from numpy.linalg import norm

from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# COMMAND ----------

# please check the folder name where file is uploaded 
folder_path = '/FileStore/tables/'
#data file
movie_metadata = sc.textFile(folder_path+'movie_metadata.tsv')
plot_summary = sc.textFile(folder_path+'plot_summaries.txt')

# input file as attached
query_string_data = sc.textFile(folder_path+'assignment_1_input-1.txt')

total_doc = plot_summary.count()

# COMMAND ----------

stop_words = stopwords.words('english')
def clean_str(strData):
    strData = ''.join([word for word in strData if word not in string.punctuation])
    strData = strData.lower().split(' ')
    strData = ' '.join([word for word in strData if word not in stop_words])
    
    return strData

# COMMAND ----------

def convert_to_vector_form(str):
    wordList = re.compile(r"\w+")
    all_applicable_words = wordList.findall(str)
    return Counter(all_applicable_words)

# COMMAND ----------

def func_tf(doc, query):
    return float(doc.split(' ').count(query)/len(doc.split(' ')))

# COMMAND ----------

def func_cosine_similarity(doc, query):
    v1 = convert_to_vector_form(doc)
    v2 = convert_to_vector_form(query)
    
    commonWords = set(v1.keys()) & set(v2.keys())
    num = sum([v1[x] * v2[x] for x in commonWords])
    
    addition1 = sum([v1[x] ** 2 for x in list(v1.keys())])
    addition2 = sum([v2[x] ** 2 for x in list(v2.keys())])
    
    denum = math.sqrt(addition1) * math.sqrt(addition2)
    
    if not denum:
        return 0.0
    else:
        return float(num) / denum


# COMMAND ----------

movie_metadata = movie_metadata.map(lambda x: ((x.split("\t"))[0], ((x.split("\t"))[2])))
plot_summary = plot_summary.map(lambda x: ((x.split("\t"))[0], clean_str((x.split("\t"))[1])))

# COMMAND ----------

query_string = query_string_data.collect()
ans = []

for str in query_string:
    search_term = clean_str(str)
    if len(str.split(' ')) == 1:
        scount = plot_summary.filter(lambda x: x[1].split(' ').count(search_term) > 0)
        ndoc = scount.count()
        idf_d = np.log(float(total_doc)/ndoc)

        movies = scount.map(lambda x: [x[0], func_tf(x[1], search_term)*idf_d]).sortBy(lambda x: -x[1]).take(10)
        
    else:
        movies = plot_summary.map(lambda x: [x[0], func_cosine_similarity(x[1], search_term)]).sortBy(lambda x: -x[1]).take(10)
        
    movie_list = sc.parallelize(movies).join(movie_metadata).sortBy(lambda x: -x[1][0]).map(lambda x: x[1][1])
    ans.append((str, movie_list.collect()))
    print("\n"+str+": ")
    print(movie_list.collect())
    

# COMMAND ----------

print(ans)

# COMMAND ----------

plot_summary.collect()

# COMMAND ----------

