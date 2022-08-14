#predict car class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load data in
df = pd.read_fwf("/Users/moksh/Downloads/car.data")
#map or introduce dummy variables
df['buying'] = df['vhigh,vhigh,2,2,small,low,unacc'].apply(lambda word:word.split(',')[0])
df['maint'] = df['vhigh,vhigh,2,2,small,low,unacc'].apply(lambda word:word.split(',')[1])
df['doors'] = df['vhigh,vhigh,2,2,small,low,unacc'].apply(lambda word:word.split(',')[2])
df['persons'] = df['vhigh,vhigh,2,2,small,low,unacc'].apply(lambda word:word.split(',')[3])
df['lugboot'] = df['vhigh,vhigh,2,2,small,low,unacc'].apply(lambda word:word.split(',')[4])
df['safety'] = df['vhigh,vhigh,2,2,small,low,unacc'].apply(lambda word:word.split(',')[5])
df['class'] = df['vhigh,vhigh,2,2,small,low,unacc'].apply(lambda word:word.split(',')[6])
df.drop("vhigh,vhigh,2,2,small,low,unacc", axis = 1, inplace = True)
g = df['class']
df['class'] = g.map({'unacc':0,'unac':0, 'acc':1, 'vgood':3, 'good':2})
dummies = pd.get_dummies(df[df.drop('class', axis = 1).columns], drop_first=True)
df.drop(df.drop('class', axis = 1).columns, axis=1, inplace = True)
df = pd.concat([df,dummies], axis = 1)

#explorartory data analysis
df['class'].plot(kind = 'hist', bins = 4)
plt.show()

#use random under sampling on data set due to data imbalance
from imblearn.under_sampling import RandomUnderSampler
sampler = RandomUnderSampler(replacement=True)
X = df.drop('class', axis = 1)
y = df['class']
X_rus, y_rus = sampler.fit_sample(X,y)

#train test split of data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=0.4)

#import sequential model, and add Dense and Dropout layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(40, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(20, activation='relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(10, activation='relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(4, activation='softmax'))
#compile modek
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam', metrics = 'accuracy')
#fit the model
model.fit(x = X_train, y = y_train, epochs = 300, validation_data=(X_test, y_test), verbose = 1)

#plot accuracy and loss
pd.DataFrame(model.history.history).plot()
plt.show()
#predictions on tets data
raw_predicts = model.predict(df.drop('class',axis = 1).values)
pred = np.argmax(raw_predicts,axis=1)

#evaluate on test data
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(df['class'].values, pred))
print(classification_report(df['class'].values, pred))






