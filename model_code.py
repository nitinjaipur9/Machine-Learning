# Importing librarries
import numpy as np
import pandas as pd
import re
import pickle

# Loading data
df = pd.read_csv('balenced_reviews.csv')

# Storing Columns in a list
column_names = df.columns.tolist()

# Sample data
sample_data = df.head()

# Removing NaN values
df.dropna(inplace=True)

# Resetting index
df = df.reset_index()

# Removing all raws where 'overall' column is 3
df = df[df['overall'] != 3]
df = df.reset_index()

# Selecting label and features
label = df['overall']
features = df['reviewText']

# Macking series of features and label
label = pd.DataFrame(label)
features = pd.DataFrame(features).iloc[:,0].values

# Cleaning data using nltk
corpus = []

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

for i in range(0, features.shape[0]):
    feature = re.sub('[^a-zA-Z]', ' ', features[i])
    feature = feature.lower()
    feature = feature.split()
    feature = [word for word in feature if not word in stopwords.words('english')]
    feature = [ps.stem(word) for word in feature]
    feature = ' '.join(feature)
    corpus.append(feature)

# Converting features to numeric form
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(min_df = 50)
features_final = cv.fit_transform(corpus)
'''
features_final = scipy.sparse.csr_matrix.sort_indices(features_final)
'''
# Saving transformer for use in model
vect = TfidfVectorizer(min_df = 50).fit(corpus)
pickle.dump(vect.vocabulary_, open('features.pkl', 'wb'))


# Train test split
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(features_final, label_final, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(feature_train, label_train)
pickle.dump(lr, open('ml_model.pkl', 'wb'))
training_score_ml = lr.score(feature_train, label_train)
testing_score_ml = lr.score(feature_test, label_test)


# Creating Deep Learning model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
node_input = len(cv.get_feature_names())

model.add(Dense(units = 7330, activation = 'relu', input_dim = 49696))
model.add(Dropout(0.2))
model.add(Dense(units = 1500, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 330, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 5, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])

import tensorflow as tf

from scipy.sparse import csr_matrix

model.fit(corpus, label_x, batch_size = 10, epochs = 5)

model.save('model.h5')