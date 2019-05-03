from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import os
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import tree
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# we're trying to train our dataset and find features that best predict if a player was/will be an all star in the nba.
# we are going to use different models for classification: decision tree, random forest, logistic regression, & XGBoost

os.chdir('/Users/drewosherow/Desktop/spring2019/itp449/python/NBA-All-Star-Classification-Models')

#open csv file
stats = pd.read_csv('Seasons_Stats 2.csv')

#store table as a dataframe
df = DataFrame(stats)
print(df.info())

#checking for null values that might skew model
null_values = df.isnull().sum()
null_values = null_values[null_values !=0].sort_values(ascending=False).reset_index()
null_values.columns = ['variable', 'number of missing']
print(null_values)

#using median value of each column to fill null values

def fillWithMedian(data):
    return data.fillna(data.median(), inplace=True)

fillWithMedian(df)
df.isnull().any()
print(df.isnull().any())
print(df.info())

#check for correlation between variables and target

corr_matrix = df.corr()
print(corr_matrix['All Star'].sort_values(ascending=False))

#split dataset into features and target variable
X = df[['Age', 'Games', 'Games Started', 'Minutes Played',
            'PER', 'TS%', '3P Attempt Rate', 'FT Rate', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%',
            'BLK%', 'TOV%', 'USG%', 'Offensive Win Shares', 'Defensive Win Shares', 'Win Shares Per 48 Minutes',
            'Win Shares Per 48 Minutes.1', 'OBPM', 'DBPM', 'BPM', 'VORP', 'FG', 'FGA', 'FG%', '3P', '3PA',
            '3P%', '2P', '2P%', 'eFG%', 'FT', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            'PTS', 'Prev All Star']]
y = df['All Star']

#split X and y into training and testing sets

#drop irrelevant variables for better model performance
# col_remove = df[['TS%', '3P Attempt Rate', 'FT Rate',
#                  'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'Defensive Win Shares',
#                  'DBPM', 'FG%', '3P%','2P%', 'eFG%', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PF',
#                  'Prev All Star']]
# X1 = X.drop(col_remove, axis=1)
# print(X1.info())

#split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

log_reg = LogisticRegression()

# fit model with data

log_reg.fit(X_train, y_train)

# make predictions
y_pred = log_reg.predict(X_test)

# evaluate performance
confusion = metrics.confusion_matrix(y_test, y_pred)
print(confusion)
print("Logistic Regression Accuracy:")
print(log_reg.score(X,y))
print(classification_report(y_test, y_pred))


## -------------------------------------------------- ##
## ---------------- RANDOM FOREST ------------------- ##

model = RandomForestClassifier(n_estimators=40, max_features=10)
model.fit(X_train, y_train)
print("Random Forest Accuracy:")
print(model.score(X_test,y_test))
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)
print(classification_report(y_test, y_pred))


## -------------------------------------------------- ##
## ---------------- XGBoost ------------------- ##

training_data = X_train
test_data = X_test

training_target = y_train
test_target = y_test

our_tree = tree.DecisionTreeClassifier()

our_tree.fit(training_data,training_target)

weak_accuracy_test = our_tree.score(test_data,test_target)

print("Decision Tree Accuracy:")
print(weak_accuracy_test)

our_xgbooster = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree= 0.25, learning_rate = 0.1,
                                  max_depth=5, alpha = 10, n_estimators=40)

our_xgbooster.fit(training_data,training_target)
strong_accuracy_test = our_xgbooster.score(test_data,test_target)
print("XGBoost Accuracy:")
print(strong_accuracy_test)


## --------------- Keras/TensorFlow Neural Network ---------------- ##

neural_data = np.loadtxt('stats_noheader.csv', delimiter=',')

# split into input (X) and output (Y) variables
X_neural = neural_data[:,0:47]
Y_neural = neural_data[:,47]

# split into 75% for train and 25% for test
X_train, X_test, y_train, y_test = train_test_split(X_neural, Y_neural, test_size=0.25, random_state=7)

# create model
model = Sequential()
model.add(Dense(12, input_dim=47, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




