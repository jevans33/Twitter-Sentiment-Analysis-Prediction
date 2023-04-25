#!/usr/bin/env python
# coding: utf-8

# In[32]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install nltk')
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install -U scikit-learn')
get_ipython().system('{sys.executable} -m pip install matplotlib')


# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# In[34]:


# Getting the dataset
fields=['target','ids','date','query','user','text']
df = pd.read_csv("tweetsentimentdata.csv", encoding="ISO-8859-1", names=fields)
df.info()


# In[35]:


pd.options.display.max_colwidth = 280


# In[36]:


#df.head(10)


# In[37]:


#df.tail(10)


# In[38]:


# Removing unnecessary columns
df = df[['target', 'text']]
#df.info()
#df.head(10)


# In[39]:


# Visualize the data sentiment distribution
sentiments = {0: "Negative", 4: "Positive"}
print(df.target.apply(lambda x: sentiments[x]).value_counts())
df.target.apply(lambda x: sentiments[x]).value_counts().plot(kind = 'bar')
plt.show()


# In[40]:


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


# In[41]:


# stopwords -> words that do not add much meaning to a sentence, such as 'the', 'have', 'she

# Remove stopwords. Using nltk library to remove stopwords
nltk.download('stopwords')

# getting stopwords
stop_words = set(stopwords.words('english'))

df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

#df.head(10)


# In[42]:


#checking list of stopwords
#do we want to add or subtract any??
#print(stopwords.words('english'))


# In[43]:


#remove punctuation
punct = r'[^\w\s]+'
df['clean_text'] = df['clean_text'].replace(to_replace=punct, value='', regex=True)

#df.head(10)


# In[44]:


#adding new stopwords
new_stopwords = ['im', 'cant']
stpwrd = nltk.corpus.stopwords.words('english')
stpwrd.extend(new_stopwords)


# In[45]:


#remove stopwords again b/c some had punctuation next to chars(...)
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stpwrd]))


# In[46]:


#replacing blank cells with NaN
df['clean_text'].replace('', np.nan, inplace=True)

#removing rows with no text
df.dropna(subset = ['clean_text'], inplace = True)


# In[47]:


#rated zeros
df.head(10)


# In[48]:


#rated 4s
df.tail(10)


# In[49]:


#from nltk.tokenize import TweetTokenizer

#tt = TweetTokenizer()
#df['token_tweet'] = df['text'].apply(tt.tokenize)
#print(df['token_tweet'])


# In[50]:


#from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')

#lemma = WordNetLemmatizer()
#df['lemmatize'] = df['token_tweet'].apply(lambda list:[lmtzr.lemmatize(word) for word in list])
#print(test['lemmatize'])


# In[51]:


df.head(10)


# In[52]:


#text, target = list(df['text']), list(df['target'])


# In[53]:


# Splitting the clean data. 90% training data, 10% test data
clean_X_train, clean_X_test, clean_y_train, clean_y_test = train_test_split(df['clean_text'], df['target'],
                                                    test_size = 0.1, random_state = 0)

#clean_X_test


# In[54]:


#clean_y_test


# In[55]:


#clean_X_train


# In[56]:


#clean_y_train


# In[57]:


# Splitting the original data. 90% training data, 10% test data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'],
                                                    test_size = 0.1, random_state = 0)


# In[65]:


# Vectorize the clean data. 
# Vectorization -> process of transforming text data into a numerical representation that can be used by models
import time
clean_t = time.time()

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(clean_X_train)
clean_X_train_vect = vectorizer.transform(clean_X_train)
clean_X_test_vect = vectorizer.transform(clean_X_test)

print(f'Time Taken: {round(time.time()-clean_t)} seconds')


# In[66]:


# Vectorize the original data. 
# Vectorization -> process of transforming text data into a numerical representation that can be used by models

t = time.time()

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(X_train)
X_train_vect = vectorizer.transform(X_train)
X_test_vect = vectorizer.transform(X_test)

print(f'Time Taken: {round(time.time()-t)} seconds')


# In[62]:


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


# In[63]:


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


# In[ ]:




