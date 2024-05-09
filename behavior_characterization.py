import pandas as pd
import os, os.path
import sys
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import time


from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

input_training_dir = "Input/Training/BCH/"
data_dir = input_training_dir + "Data/"
train_data_dir = data_dir + "TR/"
validation_data_dir = data_dir + "VAL/"

output_training_dir = "Output/Training/BCH/"
patterns_dir = output_training_dir + "Patterns/"

thresholds_computation_technique = "max"
conformance_checking_technique = "token_based"
process_discovery_technique = "inductive_miner"
noise_threshold = 0.2

@contextmanager
def suppress_stdout_stderr():
	with open(devnull, 'w') as fnull:
		with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
			yield (err, out)

def get_logs():
	n_train_logs = len([name for name in os.listdir(train_data_dir) if os.path.isfile(os.path.join(train_data_dir, name))])
	train_logs = [None] * n_train_logs
	
	for train_log_filename in os.listdir(train_data_dir):
		log = xes_importer.apply(train_data_dir + '/' + train_log_filename)
		n_log = int(train_log_filename.split("_")[-1].split(".")[0])
		train_logs[n_log] = log

	n_validation_logs = len([name for name in os.listdir(validation_data_dir) if os.path.isfile(os.path.join(validation_data_dir, name))])
	validation_logs = [None] * n_validation_logs
	
	for validation_log_filename in os.listdir(validation_data_dir):
		log = xes_importer.apply(validation_data_dir + '/' + validation_log_filename)
		n_log = int(validation_log_filename.split("_")[-1].split(".")[0])
		validation_logs[n_log] = log

	return train_logs, validation_logs


def build_petri_nets(train_logs):

	petri_nets = []
	for train_log in train_logs:
		petri_net = {}
		if process_discovery_technique == "inductive_miner":
			net, initial_marking, final_marking = inductive_miner.apply(train_log, parameters={inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: noise_threshold})
		petri_net["structure"] = net
		petri_net["initial_marking"] = initial_marking
		petri_net["final_marking"] = final_marking
		petri_nets.append(petri_net)
	return petri_nets

def write_patterns(petri_nets):
	for idx,petri_net in enumerate(petri_nets):
		pnml_exporter.apply(petri_net["structure"], petri_net["initial_marking"], patterns_dir + "PetriNet_"+str(idx)+".pnml",final_marking=petri_net["final_marking"])

def compute_thresholds(validation_logs, petri_nets):
	thresholds = []

	if(thresholds_computation_technique == "average"):
		for petri_net in petri_nets:
			threshold = 0
			for validation_log in validation_logs:
				parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
				if conformance_checking_technique == "token_based":
					log_fitness = replay_fitness.apply(validation_log, petri_net["structure"], petri_net["initial_marking"], petri_net["final_marking"], variant=replay_fitness.Variants.TOKEN_BASED)["log_fitness"]
				elif conformance_checking_technique == "alignment_based":
					aligned_traces = alignments.apply_log(validation_log, petri_net["structure"], petri_net["initial_marking"], petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_DIJKSTRA_NO_HEURISTICS)
					log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]

				threshold = threshold + log_fitness
			threshold = threshold/len(validation_logs)
			thresholds.append(threshold)
	
	elif(thresholds_computation_technique == "max"):
		for petri_net in petri_nets:
			threshold = []
			for validation_log in validation_logs:
				parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
				if conformance_checking_technique == "token_based":
					log_fitness = replay_fitness.apply(validation_log, petri_net["structure"], petri_net["initial_marking"], petri_net["final_marking"], variant=replay_fitness.Variants.TOKEN_BASED)["log_fitness"]
				elif conformance_checking_technique == "alignment_based":
					aligned_traces = alignments.apply_log(validation_log, petri_net["structure"], petri_net["initial_marking"], petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_DIJKSTRA_NO_HEURISTICS)
					log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
				threshold.append(log_fitness)
			threshold = max(threshold)
			thresholds.append(threshold)


	return thresholds

def write_thresholds(thresholds):
	thresholds_file = open(output_training_dir+"thresholds.txt","w")
	
	for idx,threshold in enumerate(thresholds):
		if(idx<len(thresholds)-1):
			thresholds_file.write("pattern_"+str(idx)+"_threshold="+str(threshold)+"\n")
		else:
			thresholds_file.write("pattern_"+str(idx)+"_threshold="+str(threshold))

	thresholds_file.close()

try:
	thresholds_computation_technique = sys.argv[1]
	conformance_checking_technique = sys.argv[2]
	process_discovery_technique = sys.argv[3]
	if process_discovery_technique == "inductive_miner":
		noise_threshold = float(sys.argv[4])
	
except IndexError:
	print("Not enough input arguments provided. Please, write the thresholds computation technique, the conformance checking technique, the process discovery technique, and the noise thresholds (if the process discovery technique is the inductive miner).")
	sys.exit()


train_logs, validation_logs = get_logs() 
petri_nets = build_petri_nets(train_logs)
thresholds = compute_thresholds(validation_logs, petri_nets)

temp_petri_nets = []
temp_thresholds = []

for idx,threshold in enumerate(thresholds):
	if threshold != 0.0:
		temp_petri_nets.append(petri_nets[idx])
		temp_thresholds.append(threshold)

petri_nets = temp_petri_nets
thresholds = temp_thresholds

write_patterns(petri_nets)
write_thresholds(thresholds)









