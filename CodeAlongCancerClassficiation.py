#Cancer classification using deep learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("DATA/cancer_classification.csv")


sns.countplot(x = 'benign_0__mal_1', data = df)
plt.show()


df.corr()['benign_0__mal_1'].sort_values().plot(kind = 'bar')
plt.show()


#train test split
X  = df.drop('benign_0__mal_1', axis = 1).values
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=101)


#Scale the data


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(X_train)

#Tensorflow imports

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

#Earky stop and tensorboard configs
early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose= 1, patience = 25)
log_directory = 'logs/fit'
timestamp = datetime.now().strftime("%Y-%m-%d--%H%M")
log_directory = log_directory + '/' + timestamp
board = TensorBoard(log_dir=log_directory, histogram_freq=1,
                   write_graph=True,
                   write_images = True,
                   update_freq='epoch',
                    profile_batch=2,
                    embeddings_freq=1
                   )


#Create the neural net


model = Sequential()

model.add(Dense(units = 30,activation='relu'))
model.add(Dropout(rate =0.5))
model.add(Dense(units = 15,activation='relu'))
model.add(Dropout(rate =0.5))
model.add(Dense(units = 1,activation='sigmoid'))

#complile neural net
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

#fit the model
model.fit(x = X_train,
          y = y_train,
          epochs = 600,
          validation_data=(X_test, y_test),
          verbose = 1,
          callbacks=[early_stop, board]
)
#evaluate the model
df = pd.DataFrame(model.history.history)
df.plot()
plt.show()
predictions = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

