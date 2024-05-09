import pandas as pd
import math
import os
import numpy as np
import sys
from random import seed
from random import random
from random import randint
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.spatial import distance

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

input_training_dir = "Input/Training/EL_EXT/"
input_training_data_dir = input_training_dir + "Data/"

output_training_dir = "Output/Training/EL_EXT/"
output_training_data_dir = output_training_dir + "Data/"
output_training_visualization_dir = output_training_dir + "Visualization/"

validation_split_percentage = -1
normal_test_split_percentage = -1
normalization_technique = "zscore"
clustering_technique = "kmeans"
n_clusters = 5
window_length = 250

@contextmanager
def suppress_stdout_stderr():
	with open(devnull, 'w') as fnull:
		with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
			yield (err, out)

def load_data():
	normal_data = pd.read_csv(input_training_data_dir + "normal_data.csv")
	anomalous_data = pd.read_csv(input_training_data_dir + "anomalous_data.csv")

	return normal_data, anomalous_data

def random_split_data(timeseries, split_percentage, window_length):

	random_windows = []
	remaining_windows = []

	n_windows = math.floor(len(timeseries)/window_length)
	n_random_windows = n_windows - math.floor(n_windows*split_percentage)
	n_remaining_windows = math.floor(n_windows*split_percentage)
	random_windows_index_pairs = []
	temp_timeseries = timeseries.copy()
	for i in range(0, n_random_windows):
		index_pair = []
		start_index = randint(0,len(temp_timeseries)-window_length)
		index_pair.append(start_index)
		index_pair.append(start_index+window_length)
		random_windows.append(temp_timeseries[index_pair[0]:index_pair[1]].reset_index(drop=True))
		temp_timeseries = temp_timeseries.drop(list(range(index_pair[0],index_pair[1])), axis=0)
		temp_timeseries.reset_index(inplace=True, drop=True)

	for i in range(0, n_remaining_windows):
		remaining_windows.append(temp_timeseries.head(window_length))
		temp_timeseries = temp_timeseries.drop(list(range(0,window_length)))
		temp_timeseries.reset_index(inplace=True, drop=True)

	return random_windows, remaining_windows

def linear_split_data(timeseries, window_length):

	linear_windows = []
	n_windows = math.floor(len(timeseries)/window_length)
	temp_timeseries = timeseries.copy()

	for i in range(0,n_windows):
		linear_windows.append(temp_timeseries.head(window_length))
		temp_timeseries = temp_timeseries.drop(range(0,window_length), axis=0)
		temp_timeseries = temp_timeseries.reset_index(drop=True)

	return linear_windows

def timeseries_split(timeseries, split_percentage):

	second_split = timeseries.copy()
	first_split = pd.DataFrame()

	n_instances_first_split = math.floor(len(timeseries)*split_percentage)
	first_split = second_split.head(n_instances_first_split)
	second_split = second_split.drop(list(range(0,n_instances_first_split)),axis=0)

	first_split = first_split.reset_index(drop=True)
	second_split = second_split.reset_index(drop=True)

	return first_split, second_split

def generate_train_test_datasets(normal_data, anomalous_data):

	test_normal, train_normal = timeseries_split(normal_data, normal_test_split_percentage)
	test_normal = linear_split_data(test_normal, window_length)
	test_anomalous = linear_split_data(anomalous_data, window_length)
	train_normal, validation_normal = random_split_data(train_normal, validation_split_percentage, window_length)

	return train_normal, validation_normal, test_normal, test_anomalous

def timestamp_builder(number):
	
	SSS = number
	ss = int(math.floor(SSS/1000))
	mm = int(math.floor(ss/60))
	hh = int(math.floor(mm/24))
	
	SSS = SSS % 1000
	ss = ss%60
	mm = mm%60
	hh = hh%24
	
	return "1900-01-01T"+str(hh)+":"+str(mm)+":"+str(ss)+"."+str(SSS)

def build_event_log(observation):
	caseid = randint(1,10000000)
	event_log = []
	previous_cluster = int(observation.iloc[0]["Cluster"])
	observation.drop(index=observation.index[0], axis=0, inplace=True)

	idx = 1
	for index, instance in observation.iterrows():
		
		current_cluster = int(instance["Cluster"])
		if current_cluster != previous_cluster:
			event_timestamp = timestamp_builder(idx+1)
			state_transition = str(previous_cluster)+"_to_"+str(current_cluster)
			event = [caseid, state_transition, event_timestamp]
			event_log.append(event)
			previous_cluster = current_cluster
			
		idx = idx+1
		
	event_log = pd.DataFrame(event_log, columns=['CaseID', 'Event', 'Timestamp'])
	event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
	event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
	event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
	event_log = log_converter.apply(event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
	return event_log

def concatenate_windows(timeseries_windows):
	temp = pd.DataFrame(columns = list(timeseries_windows[0].columns))
	for timeseries_window in timeseries_windows:
		temp = pd.concat([temp, timeseries_window], axis=0)
	temp = temp.reset_index(drop=True)
	return temp
	
def get_intervals(timeseries):

	intervals = {}
	columns = list(timeseries.columns)
	for column in columns:
		intervals[column] = [9999999999, -9999999999]
	for column in timeseries:
		temp_max = timeseries[column].max()
		temp_min = timeseries[column].min()
		if intervals[column][0] > temp_min:
			intervals[column][0] = temp_min
		if intervals[column][1] < temp_max:
			intervals[column][1] = temp_max

	return intervals

def normalize_dataset(dataset, reuse_parameters, normalization_parameters_in):
	
	normalized_dataset = dataset.copy() 
	normalization_parameters = {}

	if reuse_parameters == 0:
		if normalization_technique == "zscore":
			for column in normalized_dataset:
				column_values = normalized_dataset[column].values
				column_values_mean = np.mean(column_values)
				column_values_std = np.std(column_values)
				if column_values_std == 0:
					column_values_std = 1
				column_values = (column_values - column_values_mean)/column_values_std
				normalized_dataset[column] = column_values
				normalization_parameters[column+"_mean"] = column_values_mean
				normalization_parameters[column+"_std"] = column_values_std
		elif normalization_technique == "min-max":
			column_intervals = get_intervals(dataset)
			for column in normalized_dataset:
				column_data = normalized_dataset[column].tolist()
				intervals = column_intervals[column]
				if intervals[0] != intervals[1]:
					for idx,sample in enumerate(column_data):
						column_data[idx] = (sample-intervals[0])/(intervals[1]-intervals[0])
				normalized_dataset[column] = column_data
			for column in column_intervals:
				normalization_parameters[column+"_min"] = column_intervals[column][0]
				normalization_parameters[column+"_max"] = column_intervals[column][1]
	else:
		if normalization_technique == "zscore":
			for label in normalized_dataset:
				mean = normalization_parameters_in[label+"_mean"]
				std = normalization_parameters_in[label+"_std"]
				parameter_values = normalized_dataset[label].values
				parameter_values = (parameter_values - float(mean))/float(std)
				normalized_dataset[label] = parameter_values
		elif normalization_technique == "min-max":
			for label in normalized_dataset:
				min = normalization_parameters_in[label+"_min"]
				max = normalization_parameters_in[label+"_max"]
				parameter_values = normalized_dataset[label].values
				if min != max:
					for idx,sample in enumerate(parameter_values):
						parameter_values[idx] = (sample-min)/(max-min)
				normalized_dataset[label] = parameter_values
	
	return normalized_dataset, normalization_parameters

def cluster_dataset(dataset, reuse_parameters, clustering_parameters_in):
	clustered_dataset = dataset.copy()
	clustering_parameters = {}

	
	if reuse_parameters == 0:
		if clustering_technique == "agglomerative":
			cluster_configuration = AgglomerativeClustering(n_clusters=n_clusters, affinity='cityblock', linkage='average')
			cluster_labels = cluster_configuration.fit_predict(clustered_dataset)
		elif clustering_technique == "kmeans":
			kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(clustered_dataset)
			cluster_labels = kmeans.labels_

		clustered_dataset["Cluster"] = cluster_labels
		cluster_labels = cluster_labels.tolist()
		used = set();
		clusters = [x for x in cluster_labels if x not in used and (used.add(x) or True)]

		instances_sets = {}
		centroids = {}
		
		for cluster in clusters:
			instances_sets[cluster] = []
			centroids[cluster] = []
		
		temp = clustered_dataset
		for index, row in temp.iterrows():
			instances_sets[int(row["Cluster"])].append(row.values.tolist())
		
		n_features_per_instance = len(instances_sets[0][0])-1
		
		for instances_set_label in instances_sets:
			instances = instances_sets[instances_set_label]
			for idx, instance in enumerate(instances):
				instances[idx] = instance[0:n_features_per_instance]
			for i in range(0,n_features_per_instance):
				values = []
				for instance in instances:
					values.append(instance[i])
				centroids[instances_set_label].append(np.mean(values))
				
		clustering_parameters = centroids
			
	elif reuse_parameters == 1:
		clusters = []
		for index, instance in clustered_dataset.iterrows():
			min_value = float('inf')
			min_centroid = -1
			for centroid in clustering_parameters_in:
				centroid_coordinates = np.array([float(i) for i in clustering_parameters_in[centroid]])
				dist = np.linalg.norm(instance.values-centroid_coordinates)
				if dist<min_value:
					min_value = dist
					min_centroid = centroid
			clusters.append(min_centroid)
		
		clustered_dataset["Cluster"] = clusters
		

	return clustered_dataset, clustering_parameters

def save_log(log_nature, log_type, idx, event_log):

	xes_exporter.apply(event_log, output_training_data_dir + log_nature + "/" + log_type + "_" + str(idx) + ".xes")

	return None	

	
try:
	debug_mode = int(sys.argv[1])
	validation_split_percentage = float(sys.argv[2])
	normal_test_split_percentage = float(sys.argv[3])
	normalization_technique = sys.argv[4]
	clustering_technique = sys.argv[5]
	n_clusters = int(sys.argv[6])
	window_length = int(sys.argv[7])

except IndexError:
	print("Not enough input arguments provided. Please, enter the validation split percentage, the test split percentage, the normalization technique, the clustering technique, the number of clusters, and the window length.")
	sys.exit()

normal_data, anomalous_data = load_data()

if window_length > len(normal_data) or window_length > len(anomalous_data):
	print("The window length is longer than the length of either the normal or anomalous data")
	sys.exit()
print(normal_data)
train_normal, validation_normal, test_normal, test_anomalous = generate_train_test_datasets(normal_data, anomalous_data)


if debug_mode == 1:
	print("Number of train normal timeseries windows: " + str(len(train_normal)))
	print("Number of validation normal timeseries windows: " + str(len(validation_normal)))
	print("Number of test normal timeseries windows: " + str(len(test_normal)))
	print("Number of test anomalous timeseries windows: " + str(len(test_anomalous)))

concatenated_train_normal = concatenate_windows(train_normal)
reuse_parameters = 0
normalized_train_normal, normalization_parameters = normalize_dataset(concatenated_train_normal, reuse_parameters, None)
clustered_train_normal, clustering_parameters = cluster_dataset(normalized_train_normal, reuse_parameters, None)
reuse_parameters = 1
for idx,train_normal_timeseries in enumerate(train_normal):
	train_normal[idx], ignore = normalize_dataset(train_normal[idx], reuse_parameters, normalization_parameters)
	train_normal[idx].to_csv(output_training_visualization_dir + "TR/N_" + str(idx) + ".csv", index = False)
	train_normal[idx], ignore = cluster_dataset(train_normal[idx], reuse_parameters, clustering_parameters)
	train_normal[idx] = build_event_log(train_normal[idx])
	save_log("TR", "N", idx, train_normal[idx])
for idx,validation_normal_timeseries in enumerate(validation_normal):
	validation_normal[idx], ignore = normalize_dataset(validation_normal[idx], reuse_parameters, normalization_parameters)
	validation_normal[idx].to_csv(output_training_visualization_dir + "VAL/N_" + str(idx) + ".csv", index = False)
	validation_normal[idx], ignore = cluster_dataset(validation_normal[idx], reuse_parameters, clustering_parameters)
	validation_normal[idx] = build_event_log(validation_normal[idx])
	save_log("VAL", "N", idx, validation_normal[idx])
for idx,test_normal_timeseries in enumerate(test_normal):
	test_normal[idx], ignore = normalize_dataset(test_normal[idx], reuse_parameters, normalization_parameters)
	test_normal[idx].to_csv(output_training_visualization_dir + "TST/N_" + str(idx) + ".csv", index = False)
	test_normal[idx], ignore = cluster_dataset(test_normal[idx], reuse_parameters, clustering_parameters)
	test_normal[idx] = build_event_log(test_normal[idx])
	save_log("TST", "N", idx, test_normal[idx])
for idx,test_anomaly_timeseries in enumerate(test_anomalous):
	test_anomalous[idx], ignore = normalize_dataset(test_anomalous[idx], reuse_parameters, normalization_parameters)
	test_anomalous[idx].to_csv(output_training_visualization_dir + "TST/A_" + str(idx) + ".csv", index = False)
	test_anomalous[idx], ignore = cluster_dataset(test_anomalous[idx], reuse_parameters, clustering_parameters)
	test_anomalous[idx] = build_event_log(test_anomalous[idx])
	save_log("TST", "A", idx, test_anomalous[idx])


