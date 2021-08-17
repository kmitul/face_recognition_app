############################################ Imports ###############################################################
from deepface import DeepFace
from deepface.basemodels import VGGFace
from pathlib import Path
import numpy as np
from skimage import io 
from skimage.transform import resize
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	import keras
	from keras.models import Model, Sequential
	from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Activation, Convolution2D
elif tf_version == 2:
	from tensorflow import keras
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Activation, Convolution2D

base_dir = os.path.join(os.getcwd(),"pipeline")

############################################ Function definitions ###############################################################
# function for loading pretrained emotion detection model 
def loadModel_emotion():

	num_classes = 7

	model = Sequential()

	#1st convolution layer
	model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
	model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

	#2nd convolution layer
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	#3rd convolution layer
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	model.add(Flatten())

	#fully connected neural networks
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(num_classes, activation='softmax'))

	#----------------------------
	path = os.path.join(base_dir,"weights","facial_expression_model_weights.h5")
	model.load_weights(path)

	return model

# function for loading pretrained age approximation model 
def loadModel_age():

	model = VGGFace.baseModel()

	#--------------------------

	classes = 101
	base_model_output = Sequential()
	base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	#--------------------------

	age_model = Model(inputs=model.input, outputs=base_model_output)

	#--------------------------

	#load weights
	path = os.path.join(base_dir,"weights","age_model_weights.h5")

	age_model.load_weights(path)

	return age_model

	#--------------------------

# function for loading pretrained mask detection model 
def loadModel_mask():

	face_mask_model = load_model(os.path.join(base_dir,"weights","Xception-size-64-bs-32-lr-0.0001.h5"))

	return face_mask_model

###################################################################################################################