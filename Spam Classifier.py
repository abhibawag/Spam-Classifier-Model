# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:46:18 2021

@author: Abhishek
"""

#importing the dataset
import pandas as pd
messages = pd.read_csv('SpamClassifier-master/smsspamcollection/SMSSpamCollection', sep='\t', names = ["label", "message"])

#Data cleaning and preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
corpus=[]

for i in range (len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    #review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#Creating Bag Of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

#Train Test Splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Training Model using Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)

#Predicting from model
y_pred = spam_detect_model.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)

#printing Accuracy of model
from sklearn.metrics import  accuracy_score
accuracy = accuracy_score(y_test, y_pred)




















