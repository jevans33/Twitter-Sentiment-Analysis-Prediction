#!/usr/bin/env python
# coding: utf-8

# In[185]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install nltk')
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install -U scikit-learn')
get_ipython().system('{sys.executable} -m pip install matplotlib')
get_ipython().system('{sys.executable} -m pip install -U scattertext')
get_ipython().system('{sys.executable} -m pip install -U pip setuptools wheel')
get_ipython().system('{sys.executable} -m pip install -U spacy')
get_ipython().system('{sys.executable} -m python -m spacy download en_core_web_sm')
#!{sys.executable} -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
#!pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz


# In[198]:


get_ipython().system('{sys.executable} --version')
#!{sys.executable} spacy info


# In[192]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# In[100]:


# Getting the dataset
fields=['target','ids','date','query','user','text']
df = pd.read_csv("tweetsentimentdata.csv", encoding="ISO-8859-1", names=fields)
df.info()


# In[101]:


pd.options.display.max_colwidth = 280


# In[102]:


#df.head(10)


# In[103]:


#df.tail(10)


# In[104]:


# Removing unnecessary columns
df = df[['target', 'text']]
#df.info()
#df.head(10)


# In[105]:


# Visualize the data sentiment distribution
sentiments = {0: "Negative", 4: "Positive"}
print(df.target.apply(lambda x: sentiments[x]).value_counts())
df.target.apply(lambda x: sentiments[x]).value_counts().plot(kind = 'bar')
plt.show()


# In[106]:


# Clean the text data: 
# remove usernames, urls, hashtags, numbers, and dates
# downcast to lowercase

# patterns
url = r'https?:\/\/[\w\-\.]+\.[a-zA-Z]{2,}\/?\S*'
username = r'@\w+'
hashtag = r'#\w+'
dates = r'\d{4}-\d{2}-\d{2}'

#removing
df['clean_text'] = df['text'].replace(to_replace=r'\d+', value='', regex=True)
df['clean_text'] = df['clean_text'].replace(to_replace=url, value='', regex=True)
df['clean_text'] = df['clean_text'].replace(to_replace=username, value='', regex=True)
df['clean_text'] = df['clean_text'].replace(to_replace=hashtag, value='', regex=True)
df['clean_text'] = df['clean_text'].replace(to_replace=dates, value='', regex=True)

# lowercase text
#df['text'] = df['text'].astype('category')
df['clean_text'] = df['clean_text'].str.lower()

#df.head(10)


# In[107]:


# stopwords -> words that do not add much meaning to a sentence, such as 'the', 'have', 'she

# Remove stopwords. Using nltk library to remove stopwords
nltk.download('stopwords')

# getting stopwords
stop_words = set(stopwords.words('english'))

df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

#df.head(10)


# In[108]:


#checking list of stopwords
#do we want to add or subtract any??
#print(stopwords.words('english'))


# In[109]:


#remove punctuation
punct = r'[^\w\s]+'
df['clean_text'] = df['clean_text'].replace(to_replace=punct, value='', regex=True)

#df.head(10)


# In[110]:


#adding new stopwords
new_stopwords = ['im', 'cant']
stpwrd = nltk.corpus.stopwords.words('english')
stpwrd.extend(new_stopwords)


# In[111]:


#remove stopwords again b/c some had punctuation next to chars(...)
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stpwrd]))


# In[112]:


#replacing blank cells with NaN
df['clean_text'].replace('', np.nan, inplace=True)

#removing rows with no text
df.dropna(subset = ['clean_text'], inplace = True)


# In[113]:


#rated zeros
df.head(10)


# In[114]:


#rated 4s
df.tail(10)


# In[115]:


#tokenization

from nltk.tokenize import word_tokenize

df['token_tweet'] = df['clean_text'].apply(word_tokenize)
print(df['token_tweet'])


# In[116]:


#lemmatizer

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemma = WordNetLemmatizer()
df['lemmatize'] = df['token_tweet'].apply(lambda list:[lemma.lemmatize(word) for word in list])
print(df['lemmatize'])


# In[117]:


df.head(10)


# In[118]:


df.tail(10)


# In[119]:


#putting lemmatized data back into one string for splitting, vectorizing, & modeling

df['lemma_string'] = list(map(' '.join, df['lemmatize']))
print(df['lemma_string'])


# In[127]:


# Splitting the lemmatized data. 90% training data, 10% test data
lemma_X_train, lemma_X_test, lemma_y_train, lemma_y_test = train_test_split(df['lemma_string'], df['target'],
                                                    test_size = 0.1, random_state = 0)

#clean_X_test


# In[128]:


# Splitting the clean data. 90% training data, 10% test data
clean_X_train, clean_X_test, clean_y_train, clean_y_test = train_test_split(df['clean_text'], df['target'],
                                                    test_size = 0.1, random_state = 0)

#clean_X_test


# In[129]:


#clean_y_test


# In[130]:


#clean_X_train


# In[131]:


#clean_y_train


# In[132]:


# Splitting the original data. 90% training data, 10% test data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'],
                                                    test_size = 0.1, random_state = 0)


# In[133]:


# Vectorize the lemmatized data. 

import time
lemma_t = time.time()

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(lemma_X_train)
lemma_X_train_vect = vectorizer.transform(lemma_X_train)
lemma_X_test_vect = vectorizer.transform(lemma_X_test)

print(f'Time Taken: {round(time.time()-lemma_t)} seconds')


# In[134]:


# Vectorize the clean data. 

import time
clean_t = time.time()

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(clean_X_train)
clean_X_train_vect = vectorizer.transform(clean_X_train)
clean_X_test_vect = vectorizer.transform(clean_X_test)

print(f'Time Taken: {round(time.time()-clean_t)} seconds')


# In[135]:


# Vectorize the original data. 

t = time.time()

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(X_train)
X_train_vect = vectorizer.transform(X_train)
X_test_vect = vectorizer.transform(X_test)

print(f'Time Taken: {round(time.time()-t)} seconds')


# In[136]:


#Bernoulli on clean data

bernModel = BernoulliNB(alpha = 2.0, force_alpha = True)
bernModel.fit(clean_X_train_vect, clean_y_train)

# Predict values for Test dataset
clean_y_pred = bernModel.predict(clean_X_test_vect)

# Print the evaluation metrics for the dataset.
print(classification_report(clean_y_test, clean_y_pred))
    
# Compute and plot the Confusion matrix
cf_matrix = confusion_matrix(clean_y_test, clean_y_pred, labels=bernModel.classes_)
print(cf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = bernModel.classes_)
disp.plot()
plt.show()


# In[137]:


#Bernoulli on original data

bernModel = BernoulliNB(alpha = 2.0, force_alpha = True)
bernModel.fit(X_train_vect, y_train)

# Predict values for Test dataset
y_pred = bernModel.predict(X_test_vect)

# Print the evaluation metrics for the dataset.
print(classification_report(y_test, y_pred))
    
# Compute and plot the Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred, labels=bernModel.classes_)
print(cf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = bernModel.classes_)
disp.plot()
plt.show()


# In[138]:


#Bernoulli on lemmatized data

bernModel = BernoulliNB(alpha = 2.0, force_alpha = True)
bernModel.fit(lemma_X_train_vect, lemma_y_train)

# Predict values for Test dataset
lemma_y_pred = bernModel.predict(lemma_X_test_vect)

# Print the evaluation metrics for the dataset.
print(classification_report(lemma_y_test, lemma_y_pred))
    
# Compute and plot the Confusion matrix
cf_matrix = confusion_matrix(lemma_y_test, lemma_y_pred, labels=bernModel.classes_)
print(cf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = bernModel.classes_)
disp.plot()
plt.show()


# In[ ]:


#Helpful references

#NTLK tokenize:  https://www.nltk.org/api/nltk.tokenize.casual.html

#Stemming and lemmatization in PythonNLTK with examples: 
# https://www.guru99.com/stemming-lemmatization-python-nltk.html  

#Text preprocessing with NLTK: https://towardsdatascience.com/text-preprocessing-with-nltk-9de5de891658#14f5


# In[188]:


import scattertext as st
import spacy
from spacy.lang.en.examples import sentences
import en_core_web_sm

#nlp = en_core_web_sm.load()
nlp = en_core_web_sm.load()

nlp = spacy.load("en_core_web_sm")


corpus = st.CorpusFromPandas(df, category_col='target', text_col='lemmatize',  nlp=nlp).build()
sent = st.produce_scattertext_explorer(corpus,

        category='negative',

        category_name='Negative',

        not_category_name='Positive',

        width_in_pixels=1000,

        metadata=df['lemmatize'])


# In[ ]:


open(“Twitter_Sentiment.html", 'wb').write(html.encode('utf-8'))

