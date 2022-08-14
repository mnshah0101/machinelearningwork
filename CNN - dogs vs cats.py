#imports
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

#creates directory variable for Cat and Dog folders
DIRECTORY = r'/Users/moksh/Downloads/kagglecatsanddogs_5340/PetImages'
#creates categoires variable for each file path
CATEGORIES = ['Cat', 'Dog']

#image size (100,100)
img_size = 100
 #data list
data = []

#training data list
training_data=[]

#function which creates training data
def create_training_data():
    for category in CATEGORIES: #for each category in the list- cat and then dog
        path=os.path.join(DIRECTORY, category) #creates a new path, based on directory and then either cat or dog
        class_num=CATEGORIES.index(category) #the class num, which is either 0 or 1
        for img in os.listdir(path): #for each image in the path
            try:
                img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #create the image array
                new_array=cv2.resize(img_array,(img_size,img_size)) #resize the image array
                training_data.append([new_array,class_num]) #add it to training data
            except Exception as e:
                pass
create_training_data() #call create training data function
print(len(training_data)) #print length of training data
random.shuffle(data) #shuffle data
X=[]
y=[]

for categories, label in training_data: #for each category, and label, add to the lists
    X.append(categories)
    y.append(label)
X= np.array(X) #make them arrays
y = np.array(y)

X= np.array(X).reshape(-1, 100, 100, 1) #reshape so it becomes 25000 images that are 100x100x1
#featuring, 255 is the highest value
X = X/255

from tensorflow.keras.models import Sequential #create a sequential model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #import layers

model = Sequential() #create an instance of the sequential model
model.add(Conv2D(64, (3,3), activation = "relu") ) #create a Conv2d layer, with the following dimensions
model.add(MaxPooling2D((2,2)) )

model.add(Conv2D(64, (3,3), activation = "relu") ) #add another conv2d layer
model.add(MaxPooling2D((2,2)) )

model.add(Flatten()) #flattens the model

model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu')) #25000 images that are 100x100x1
model.add(Dense(2, activation = 'softmax')) #cat or dog

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "adam", metrics = ['accuracy']) #use the categorical cross entropy loss function, using the adam optimizer, and tracks the accuracy metric

model.fit(X,y, epochs = 5, validation_split = 0.1, verbose = 1) #fithe the model

loss_df = pd.DataFrame(model.history.history) #plot the model's accuracy and loss
loss_df.plot()
plt.show()


