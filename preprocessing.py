#!/usr/bin/env python
# coding: utf-8

# In[144]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install nltk')
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install -U scikit-learn')
get_ipython().system('{sys.executable} -m pip install matplotlib')


# In[187]:


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


# In[146]:


# Getting the dataset
fields=['target','ids','date','query','user','text']
df = pd.read_csv("tweetsentimentdata.csv", encoding="ISO-8859-1", names=fields)
df.info()


# In[147]:


pd.options.display.max_colwidth = 280


# In[148]:


#df.head(10)


# In[149]:


#df.tail(10)


# In[150]:


# Removing unnecessary columns
df = df[['target', 'text']]
#df.info()
#df.head(10)


# In[151]:


# Visualize the data sentiment distribution
sentiments = {0: "Negative", 4: "Positive"}
print(df.target.apply(lambda x: sentiments[x]).value_counts())
df.target.apply(lambda x: sentiments[x]).value_counts().plot(kind = 'bar')
plt.show()


# In[152]:


# Clean the text data: 
# remove usernames, urls, hashtags, numbers, and dates
# downcast to lowercase

# patterns
url = r'https?:\/\/[\w\-\.]+\.[a-zA-Z]{2,}\/?\S*'
username = r'@\w+'
hashtag = r'#\w+'
dates = r'\d{4}-\d{2}-\d{2}'

#removing
df['text'] = df['text'].replace(to_replace=r'\d+', value='', regex=True)
df['text'] = df['text'].replace(to_replace=url, value='', regex=True)
df['text'] = df['text'].replace(to_replace=username, value='', regex=True)
df['text'] = df['text'].replace(to_replace=hashtag, value='', regex=True)
df['text'] = df['text'].replace(to_replace=dates, value='', regex=True)

# lowercase text
#df['text'] = df['text'].astype('category')
df['text'] = df['text'].str.lower()

#df.head(10)


# In[153]:


# stopwords -> words that do not add much meaning to a sentence, such as 'the', 'have', 'she

# Remove stopwords. Using nltk library to remove stopwords
nltk.download('stopwords')

# getting stopwords
stop_words = set(stopwords.words('english'))

df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

#df.head(10)


# In[154]:


#checking list of stopwords
#do we want to add or subtract any??
#print(stopwords.words('english'))


# In[155]:


#remove punctuation
punct = r'[^\w\s]+'
df['text'] = df['text'].replace(to_replace=punct, value='', regex=True)

#df.head(10)


# In[156]:


#adding new stopwords
new_stopwords = ['im', 'cant']
stpwrd = nltk.corpus.stopwords.words('english')
stpwrd.extend(new_stopwords)


# In[157]:


#remove stopwords again b/c some had punctuation next to chars(...)
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stpwrd]))


# In[158]:


#replacing blank cells with NaN
df['text'].replace('', np.nan, inplace=True)

#removing rows with no text
df.dropna(subset = ['text'], inplace = True)


# In[159]:


#rated zeros
#df.head(10)


# In[160]:


#rated 4s
#df.tail(10)


# In[161]:


#text, target = list(df['text']), list(df['target'])


# In[162]:


# Splitting the data. 90% training data, 10% test data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'],
                                                    test_size = 0.1, random_state = 0)

X_test


# In[163]:


y_test


# In[164]:


X_train


# In[165]:


y_train


# In[166]:


# Vectorize the data. 
# Vectorization -> process of transforming text data into a numerical representation that can be used by models
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)


# In[189]:


bernModel = BernoulliNB(alpha = 2.0, force_alpha = True)
bernModel.fit(X_train, y_train)

# Predict values for Test dataset
y_pred = bernModel.predict(X_test)

# Print the evaluation metrics for the dataset.
print(classification_report(y_test, y_pred))
    
# Compute and plot the Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred, labels=bernModel.classes_)
print(cf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = bernModel.classes_)
disp.plot()
plt.show()


# In[ ]:




