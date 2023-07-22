# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 19:47:51 2023

@author: shivu
"""

import pandas as pd

df = pd.read_csv('emails.csv')
df.head()


from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('emails.csv')

# Exclude non-numeric and target column from features
X = df.drop(['Email No.', 'spam'], axis=1)

# Separate the target variable (y)
y = df['spam']

# Create a RandomForestClassifier model
model = RandomForestClassifier()

# Fit the model to the data to calculate feature importances
model.fit(X, y)

# Get feature importances from the model
importances = model.feature_importances_

# Create a DataFrame to store feature importances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the features by importance in descending order
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# Select the top 1.5K features
selected_features = feature_importances.head(1500)['Feature'].tolist()

# Create a new DataFrame with only the selected features
reduced_df = df[['Email No.'] + selected_features]

# Print the reduced DataFrame
print(reduced_df.head())

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Separate the features (X) and the target variable (y)
X = df['Email No.']
y = df['spam']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Create and train the Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Vectorize the test data using the same vectorizer
X_test_vectorized = vectorizer.transform(X_test)

# Make predictions on the test data
y_pred = clf.predict(X_test_vectorized)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns


# Select five words of choice
selected_words = ['free', 'urgent', 'money', 'offer', 'win']

# Extract the features (X) and the target variable (y)
X = df['Email No.']
y = df['spam']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer with selected words
vectorizer = CountVectorizer(vocabulary=selected_words)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train the Support Vector Classifier (SVC)
svc = SVC()
svc.fit(X_train_vectorized, y_train)

# Make predictions on the test data
y_pred = svc.predict(X_test_vectorized)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate the Confusion Matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)

# Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

