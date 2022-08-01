#imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#data
df = pd.read_csv('/Users/moksh/Desktop/pythonProject/iris.csv')

#data pre-processing
from sklearn.model_selection import train_test_split
X = df.drop('target', axis = 1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model =Sequential()
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(units = 3, activation = 'softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x = X_train, y = y_train, epochs = 200, validation_data = (X_test, y_test), verbose = 1)

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

predictions = model.predict(X_test)
print(predictions)
predictions = np.argmax(predictions, axis=1)
print(predictions)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(model.evaluate(X_test, y_test))

