#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import re
get_ipython().system('pip install wordcloud')


import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# In[7]:


nltk.download('punkt')
nltk.download('wordnet')
def preprocess_df(textDF):
    
    # patterns
    url = r'https?:\/\/[\w\-\.]+\.[a-zA-Z]{2,}\/?\S*'
    username = r'@\w+'
    hashtag = r'#\w+'
    dates = r'\d{4}-\d{2}-\d{2}'
    
    lemmatizer = WordNetLemmatizer()
    
    # stopwords
    stop_words = ['a', 'about', "I'm" 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further','he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', "I'm", "i'm" 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
    
    textDF = textDF.replace(to_replace=r'\d+', value='', regex=True)
    textDF = textDF.replace(to_replace=url, value='', regex=True)
    textDF = textDF.replace(to_replace=username, value='', regex=True)
    textDF = textDF.replace(to_replace=hashtag, value='', regex=True)
    textDF = textDF.replace(to_replace=dates, value='', regex=True)
    
    textDF = textDF.str.lower()
    
    # remove stopwords and lemmantize
    textDF = textDF.apply(lambda x: [lemmatizer.lemmatize(word) for word in word_tokenize(x) if word.lower() not in stop_words and len(word) > 1])
    # join the lemmatized words back into a string
    textDF = textDF.apply(lambda x: ' '.join(x))
    
    #remove punctuation
    punct = r'[^\w\s]+'
    textDF = textDF.replace(to_replace=punct, value='', regex=True)
    
    return textDF

        
        


# In[8]:


# Getting the dataset
fields=['target','ids','date','query','user','text']
df = pd.read_csv("tweetsentimentdata.csv", encoding="ISO-8859-1", names=fields)


# In[9]:


pd.options.display.max_colwidth = 280
# Removing unnecessary columns
df = df[['target', 'text']]


# In[10]:


df['text'] = preprocess_df(df['text'])

#replacing blank cells with NaN
df['text'].replace('', np.nan, inplace=True)

#removing rows with no text
df.dropna(subset = ['text'], inplace = True)


# In[11]:


# save preprocessed data to new csv file
df.to_csv("preprocessed_tweetsentimentdata.csv", index = False)

