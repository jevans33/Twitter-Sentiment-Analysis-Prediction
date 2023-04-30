# Twitter-Sentiment-Analysis-Prediction

## Project Overview
The Twitter Sentiment Analysis Project is a project for CAP4770. The goal of this project is to develop a model for analyzing tweet sentiments and make accurate predictions of the positivity, negativity, or neutrality of the tweet based on specific words and their meaning/context. The project involves analyzing the dataset of tweets offered by Kaggle, "Sentiment140 dataset with 1.6 million tweets", preprocessing the text data, training three different to predict sentiment, and visualizing the results.

## Project Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- seaborn
- wordcloud

## Dataset
For this project, the dataset to be analyzed is Kaggle’s, “Sentiment140 dataset with 1.6 million tweets”. This dataset, as the name suggests, contains 1.6 million tweets that were extracted using the Twitter API. Each row in the dataset represents a tweet. Each tweet has 6 fields related to it, but for this project the two most important fields are the target field and text field. The target field can have three possible values: 0 if the tweet is negative, and 4 if positive. The text field contains the tweet itself. With this information it is possible to analyze the dataset and train a model based on the tweet texts and target score to make a prediction on whether a new tweet is positive, neutral or negative.


## Data Preprocessing
The text data in the dataset is preprocessed before being used to train the machine learning model. The preprocessing steps include:
- Removing URLs, usernames, hashtags, and dates
- Converting text to lowercase
- Removing stopwords
- Lemmatizing words
- Removing punctuation

## Models
The models used in this project are Logistic Regression, Bernoulli Naive Bayes and Support Vector Machines (SVM) models trained on a bag-of-words representation of the preprocessed text data. The models are trained on a subset of the dataset and evaluated on a holdout set to measure its accuracy in predicting sentiment.

## Data Visualization
The results of the sentiment analysis are visualized using a word cloud, which displays the most common words in the positive and negative tweets. The word cloud is created using the Python library `wordcloud`. A confusion matrix was also used to visualize the metrics of each model
