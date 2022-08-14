#Kaggle Playground Dataset- unsupervised machine learning competition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#read in data
df = pd.read_csv("/Users/moksh/Downloads/tabular-playground-series-jul-2022/data.csv")
df = df.drop('id', axis = 1)

#data analysis, find which columns are distributed normally
df.hist(figsize=(12,12))
plt.show()


#use a power transformer to make data more Gaussain like
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer()
df_transformed = scaler.fit_transform(df)


#plot histograms for transformed dataframe
df_transformed = pd.DataFrame(data = df_transformed)
df_transformed.hist(figsize=(12,12))

#use a BayesianGaussainMixture clustering model, since columns are in a Gaussain dist.
from sklearn.mixture import BayesianGaussianMixture
gm = BayesianGaussianMixture(
n_components=7,
verbose=1,
n_init =10,
verbose_interval=100,
random_state=2
)


#find probabliites of data being in each cluster
predicted = gm.fit_predict(df_transformed)
probs = gm.predict_proba(df_transformed)


#add these to the dataframe
df_transformed = pd.DataFrame(df_transformed)
df_transformed['class'] = predicted
df_transformed['max probability'] = np.max(probs, axis = 1)

#create a test, train split on the rows with the highest probability
threshold = 0.8
train_split = 0.8
classification_data = df_transformed[df_transformed['max probability']>threshold]
X_train = df_transformed.head(n = int(train_split*len(classification_data)))
y_train = X_train['class'].values
X_train = X_train.drop(['class', 'max probability'], axis = 1).values
X_test = df_transformed.tail(n = int((1-train_split)*len(classification_data) + 1))
y_test = X_test['class'].values
X_test = X_test.drop(['class', 'max probability'], axis = 1).values


#train another Gaussain classifier on the data points with the highest probability of classification
gm = BayesianGMMClassifier(n_components=7,covariance_type='full',max_iter= 500, init_params='k-means++', verbose=1, n_init =10, verbose_interval=100)
gm.fit(X_train,y_train)
print('Done with fitting')
y_pred = gm.predict(X_test)
print('Done with prediction')
print(accuracy_score(y_test,y_pred))

#make final predictions using this model
final_predicitons = gm.predict(df_transformed.drop(columns = ['class', 'max probability']))
#save the results in a csv
ss = pd.read_csv("/Users/moksh/Desktop/pythonProject/sample_submission.csv")
ss["Predicted"] = final_predicitons
ss.to_csv("KaggleSubmission.csv",index = False)

