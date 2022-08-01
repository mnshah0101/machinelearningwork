import pandas as pd
import numpy as np
from sklego.mixture import BayesianGMMClassifier
from sklearn.metrics import accuracy_score

df_transformed = pd.read_csv("/Users/moksh/Desktop/pythonProject/data_preprocessed.csv")
threshold = 0.8
train_split = 0.8
classification_data = df_transformed[df_transformed['max probability']>threshold]
X_train = df_transformed.head(n = int(train_split*len(classification_data)))
y_train = X_train['class'].values
X_train = X_train.drop(['class', 'max probability'], axis = 1).values
X_test = df_transformed.tail(n = int((1-train_split)*len(classification_data) + 1))
y_test = X_test['class'].values
X_test = X_test.drop(['class', 'max probability'], axis = 1).values



gm = BayesianGMMClassifier(n_components=7,covariance_type='full',max_iter= 500, init_params='k-means++', verbose=1, n_init =10, verbose_interval=100)
gm.fit(X_train,y_train)
print('Done with fitting')
y_pred = gm.predict(X_test)
print('Done with prediction')
print(accuracy_score(y_test,y_pred))

final_predicitons = gm.predict(df_transformed.drop(columns = ['class', 'max probability']))

ss = pd.read_csv("/Users/moksh/Desktop/pythonProject/sample_submission.csv")
ss["Predicted"] = final_predicitons
ss.to_csv("KaggleSubmission.csv",index = False)

