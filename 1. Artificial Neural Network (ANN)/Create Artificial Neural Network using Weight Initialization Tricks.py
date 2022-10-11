import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]


# Create dummy variables
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

## Concat the data frame
X = pd.concat([X, geography, gender], axis=1)

# Drop unnecessary columns
X = X.drop(['Geography', 'Gender'], axis=1)


# Spliting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scalling (for taking the inputs in the same scale)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=10, kernel_initializer='he_normal', activation='relu', input_dim=11))
# Adding Dropout
classifier.add(Dropout(0.3))

# Adding the second hidden layer
classifier.add(Dense(units=20, kernel_initializer='he_normal', activation='relu'))
# Adding Dropout
classifier.add(Dropout(0.4))

# Adding the third hidden layer
classifier.add(Dense(units=15, kernel_initializer='he_normal', activation='relu'))
# Adding Dropout
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)




y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)




























