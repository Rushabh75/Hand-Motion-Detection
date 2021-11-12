import os
import numpy as np
import pandas as pd
import cv2 as cv2
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow import keras


from keras.applications.vgg16 import VGG16

# We need to get all the paths for the images to later load them
# imagepaths = []
#
# # Go through all the files and subdirectories inside a folder and save path to images inside list
# for root, dirs, files in os.walk(".", topdown=False):
#   for name in files:
#     path = os.path.join(root, name)
#     if path.endswith("png"): # We want only the images
#       imagepaths.append(path)
#
# print(len(imagepaths)) # If > 0, then a PNG image was loaded
#
# print(imagepaths[0])
#
# X = [] # Image data
# y = [] # Labels
#
# # Loops through imagepaths to load images and labels into arrays
# for path in imagepaths:
#   img = cv2.imread(path) # Reads image and returns np.array
#   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
#   img = cv2.resize(img, (256, 256)) # Reduce image size so training can be faster
#   X.append(img)
#
#   # Processing label in image path
#   category = path.split("\\")[4]
#   label1 = category.split("_")[6] # We need to convert 10_down to 00_down, or else it crashes
#   if label1 == "Right Swipe" or label1 == "Right":
#     y.append(0)
#   elif label1 == "Left Swipe" or label1 == "Left":
#     y.append(1)
#   else:
#     y.append(2)
#
# #Turn X and y into np.array to speed up train_test_split
# X = np.array(X, dtype="uint8")
# X = X.reshape(len(imagepaths), 256, 256, 1) # Needed to reshape so CNN knows it's different images
# y = np.array(y)
#
# print("Images loaded: ", len(X))
# print("Labels loaded: ", len(y))
#
# print(y[0], imagepaths[0]) # Debugging#
#
# from sklearn.model_selection import train_test_split # Helps with organizing data for training
# from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix
# ts = 0.3 # Percentage of images that we want to use for testing. The rest is used for training.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
#

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

# model = Sequential()
# model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(256, 256, 1)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', # Optimization routine, which tells the computer how to adjust the parameter values to minimize the loss function.
#               loss='sparse_categorical_crossentropy', # Loss function, which tells us how bad our predictions are.
#               metrics=['accuracy']) # List of metrics to be evaluated by the model during training and testing.

model = keras.models.load_model('handrecognition_model.h5')
model.summary()

#model.fit(X_train, y_train, epochs=5,validation_data=(X_test, y_test))

#model.save('handrecognition_model.h5')

classes = [
    "Swiping Right",
    "Swiping Left",
    "Zoom In"
    ]

import cv2
def normaliz_data(np_data):
  # Normalisation
  scaler = StandardScaler()
  # scaled_images  = normaliz_data2(np_data)

  return scaled_images
to_predict = []

num_frames = 0
cap = cv2.VideoCapture(0)
classe =''

while(True):
    # Capture frame-by-frame
  ret,frame = cap.read()

    # Our operations on the frame come here
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  to_predict.append(cv2.resize(gray, (256, 256)))
  # print(gray.shape)
  # print(to_predict[0].shape)
  if len(to_predict) == 30:
    #converted_img = Image.open(current_img).convert('L')
    frame_to_predict = np.array(to_predict, dtype="uint8")
    #frame_to_predict = normaliz_data(frame_to_predict)
        #print(frame_to_predict)
    for ele in frame_to_predict:
      scaled_images = ele.reshape(-1, 256, 256,1)
      predict = model.predict(scaled_images)
      #print(classes[predict])
      classe = classes[np.argmax(predict)]


      print('Class = ',classe, 'Precision = ', np.argmax(predict)*100,'%')


        #print(frame_to_predict)
    to_predict = []
        #sleep(0.1) # Time in seconds
        #font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),1,cv2.LINE_AA)


    # Display the resulting frame
  cv2.imshow('Hand Gesture Recognition',frame)
  if cv2.waitKey(100) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()


def normaliz_data(np_data):
  # Normalisation
  scaler = StandardScaler()
  # scaled_images  = normaliz_data2(np_data)
  scaled_images = np_data.reshape(256,256)
  return scaled_images
