import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from matplotlib import ticker

from tensorflow.keras.layers import Dense,Input,Concatenate
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

import sys

from random import randint

input_training_dir = "Input/Training/DR/"
input_training_data_dir = input_training_dir + "Data/"

output_training_dir = "Output/Training/DR/"
output_training_data_dir = output_training_dir + "Data/"
output_training_model_dir = output_training_dir + "Model/"

hidden_neurons = 100
latent_code_dimension = 2
epochs = 150
debug_mode = -1

def autoencoder(hidden_neurons,latent_code_dimension):
	input_layer = Input(shape=(11,)) # Input
	encoder = Dense(hidden_neurons,activation="relu")(input_layer) # Encoder
	code = Dense(latent_code_dimension)(encoder) # Code
	decoder = Dense(hidden_neurons,activation="relu")(code) # Decoder
	output_layer = Dense(11,activation="linear")(decoder) # Output
	model = Model(inputs=[input_layer],outputs=[output_layer])
	model.compile(optimizer="adam",loss="mse")
	if debug_mode == 1:
		model.summary()
	return model

def get_compressed_data(normal_data, anomalous_data, model, latent_code_dimension):
	get_code = K.function([model.layers[0].input],[model.layers[2].output])
	columns = []
	for i in range(0,latent_code_dimension):
		columns.append("f_" + str(i))
	normal_compressed_data = pd.DataFrame(get_code(np.array(normal_data))[0], columns=columns)
	anomalous_compressed_data = pd.DataFrame(get_code(np.array(anomalous_data))[0], columns=columns)
	if debug_mode == 1:
		print("Dimensionality reduction info:")
		print("Compressed feature space: " + str(latent_code_dimension))
	
	return normal_compressed_data, anomalous_compressed_data

def load_data():
	
	normal_data = pd.read_csv(input_training_data_dir + "normal_processed_data.csv")
	anomalous_data = pd.read_csv(input_training_data_dir + "anomalous_processed_data.csv")
	
	normal_data = normal_data.drop(labels = ["labels"], axis=1)
	anomalous_data = anomalous_data.drop(labels = ["labels"], axis=1)

	return normal_data, anomalous_data


def train_autoencoder(normal_data, ae_test_split_percentage, ae_validation_split_percentage, hidden_neurons, latent_code_dimension, epochs):

	train_normal,test_normal = train_test_split(normal_data,test_size=ae_test_split_percentage,shuffle=True)
	train_data = np.array(train_normal)
	test_normal = np.array(test_normal)
	assert latent_code_dimension < 11, print("Il codice dell'autoencoder deve essere strettamente minore del numero di features")
	model = autoencoder(hidden_neurons,latent_code_dimension)
	if debug_mode == 1:
		history = model.fit(train_normal,train_normal,epochs=epochs,shuffle=True,verbose=1,validation_split=ae_validation_split_percentage,callbacks= [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'), ModelCheckpoint(output_training_model_dir + "dr_model.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
	else:
		history = model.fit(train_normal,train_normal,epochs=epochs,shuffle=True,verbose=0,validation_split=ae_validation_split_percentage,callbacks= [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'), ModelCheckpoint(output_training_model_dir + "dr_model.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=0)])

	return model, train_data

def save_compressed_dataset(codes_normal, codes_anomalous):

	codes_normal.to_csv(output_training_data_dir + "normal_data.csv", index=False)
	codes_anomalous.to_csv(output_training_data_dir + "anomalous_data.csv", index=False)

	return None

try:
	debug_mode = int(sys.argv[1])
	ae_test_split_percentage = float(sys.argv[2])
	ae_validation_split_percentage = float(sys.argv[3])
	
except IndexError:
	print("Not enough input arguments. Please, specify the debug mode and the test split percentage for the autoencoder")
	sys.exit()


normal_processed_data, anomalous_processed_data = load_data()
model, train_data = train_autoencoder(normal_processed_data, ae_test_split_percentage, ae_validation_split_percentage, hidden_neurons, latent_code_dimension, epochs)
normal_compressed_data, anomalous_compressed_data = get_compressed_data(normal_processed_data, anomalous_processed_data, model, latent_code_dimension)
save_compressed_dataset(normal_compressed_data, anomalous_compressed_data)

'''
train_normal, validation_normal, test_normal, test_anomaly = generate_train_test_datasets(codes_normal, codes_anomalous)
print("Number of train normal timeseries windows: " + str(len(train_normal)))
print("Number of validation normal timeseries windows: " + str(len(validation_normal)))
print("Number of test normal timeseries windows: " + str(len(test_normal)))
print("Number of test anomalous timeseries windows: " + str(len(test_anomaly)))
save_timeseries_windows(train_normal, validation_normal, test_normal, test_anomaly)
'''
