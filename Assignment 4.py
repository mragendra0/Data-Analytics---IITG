# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("car_evaluation.csv")
print(df.head())
print(df.describe())
# view dimensions of dataset
print(df.shape)
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
col_names
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
for col in col_names:
    print(df[col].value_counts())

# Check for missing values
print(df.isnull().sum())

#feature vector and target variable
X = df.drop(['class'], axis=1)

y = df['class']

#  split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# Check the shape of X_train and X_test
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Check data types in X_train
print(X_train.dtypes)

print(X_train.head())

X_train = X_train.apply(LabelEncoder().fit_transform)
X_test = X_test.apply(LabelEncoder().fit_transform)

print(X_train.head())



from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.feature_selection import RFE

def score(model, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
#     print(confusion_matrix(y_test, preds))
    accuracy = round(accuracy_score(y_test, preds), 5)
    print('Accuracy for', title, ':', accuracy, '\n')

lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
score(lg, "Logistic Regression")

classifiers = [
    LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
    AdaBoostClassifier(n_estimators=1000, random_state=0),
    RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
]

# Train and evaluate each classifier
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    accuracy = round(accuracy_score(y_test, preds), 5)
    print(classifier.__class__.__name__ + " Accuracy:", accuracy)
    
    
# Define the classifiers
classifiers = [
    GaussianNB(),
    KNeighborsClassifier(),
    DecisionTreeClassifier()
]

# Train and evaluate each classifier
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    accuracy = round(accuracy_score(y_test, preds), 5)
    print(classifier.__class__.__name__ + " Accuracy:", accuracy)
    

plt.figure(figsize=(200,30))








