#predict whether college is private or public with clustering algo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Get data
df = pd.read_csv("College_Data")


#Exploratory Data Analysis


sns.scatterplot(x = "Room.Board", y = "Grad.Rate", data = df, hue = "Private")
plt.show(



sns.scatterplot(x = "Outstate", y = "F.Undergrad", data = df, hue = "Private")
plt.show()


plt.figure(figsize = (10,10))
df[df["Private"] == "No"]["Outstate"].plot(kind = "hist", alpha = 0.5, bins = 20, label = "Public")
df[df["Private"] == "Yes"]["Outstate"].plot(kind = "hist", alpha = 0.5, bins = 20, label = "Private")
plt.legend()
plt.show()


plt.figure(figsize = (10,10))
df[df["Private"] == "No"]["Grad.Rate"].plot(kind = "hist", alpha = 0.5, bins = 20, label = "Public")
df[df["Private"] == "Yes"]["Grad.Rate"].plot(kind = "hist", alpha = 0.5, bins = 20, label = "Private")
plt.legend()
plt.show()


# Fix bad data


df['Grad.Rate'] = df['Grad.Rate'].replace(118,100)



plt.figure(figsize = (10,10))
df[df["Private"] == "No"]["Grad.Rate"].plot(kind = "hist", alpha = 0.5, bins = 20, label = "Public")
df[df["Private"] == "Yes"]["Grad.Rate"].plot(kind = "hist", alpha = 0.5, bins = 20, label = "Private")
plt.legend()
plt.show()


# ## K Means Cluster Creation

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X = df.drop(["Private","Unnamed: 0"], axis = 1))
clusters = kmeans.cluster_centers


# Evaluation


def converter(cluster):
    if cluster == "Yes":
        return 0
    if cluster == "No":
        return 1
df['Cluster'] = df["Private"].apply(converter)

df['Predictedlabels'] = kmeans.labels_


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['Cluster'],df["Predictedlabels"]))
print(classification_report(df['Cluster'],df["Predictedlabels"]))

