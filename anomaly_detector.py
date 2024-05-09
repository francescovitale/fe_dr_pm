import os, os.path
from os import devnull
from contextlib import contextmanager,redirect_stderr,redirect_stdout
import time
import sys

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

input_testing_dir = "Input/Testing/AD/"
input_data_dir = input_testing_dir + "Data/"
input_patterns_dir = input_testing_dir + "Patterns/"

output_testing_dir = "Output/Testing/AD/"


@contextmanager
def suppress_stdout_stderr():
	"""A context manager that redirects stdout and stderr to devnull"""
	with open(devnull, 'w') as fnull:
		with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
			yield (err, out)

def get_data():

	n_test_logs = len([name for name in os.listdir(input_data_dir) if os.path.isfile(os.path.join(input_data_dir, name)) and name.split("_")[0]=="N"])
	n_anomalous_logs = len([name for name in os.listdir(input_data_dir) if os.path.isfile(os.path.join(input_data_dir, name)) and name.split("_")[0]=="A"])

	test_logs = [None] * n_test_logs
	anomalous_logs = [None] * n_anomalous_logs

	for log_file in os.listdir(input_data_dir):
		filepath = os.path.join(input_data_dir, log_file)
		n_log = int(log_file.split("_")[-1].split(".")[0])
		if log_file.split("_")[0] == "N":
			test_logs[n_log] = xes_importer.apply(filepath)
		elif log_file.split("_")[0] == "A":
			anomalous_logs[n_log] = xes_importer.apply(filepath)

	return anomalous_logs, test_logs

def get_patterns():
	
	n_petri_nets = len([name for name in os.listdir(input_patterns_dir) if os.path.isfile(os.path.join(input_patterns_dir, name))])
	petri_nets = [None] * n_petri_nets
	for petri_net_file in os.listdir(input_patterns_dir):
		filepath = os.path.join(input_patterns_dir, petri_net_file)
		petri_net = {}
		n_net = int(petri_net_file.split("_")[-1].split(".")[0])
		net, initial_marking, final_marking = pnml_importer.apply(filepath)
		petri_net["structure"] = net
		petri_net["initial_marking"] = initial_marking
		petri_net["final_marking"] = final_marking
		petri_nets[n_net] = petri_net

	return petri_nets

def get_thresholds():
	thresholds = []
	thresholds_file = open(input_testing_dir+"thresholds.txt","r")
	readlines = thresholds_file.readlines()
	for line in readlines:
		thresholds.append(float(line.split("=")[1]))
	thresholds_file.close()
	return thresholds

def classify_logs(anomalous_logs, test_logs, normal_petri_nets, thresholds):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for test_log in test_logs:
		for idx,normal_petri_net in enumerate(normal_petri_nets):
			parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
			if conformance_checking_technique == "token_based":
				log_fitness = replay_fitness.apply(test_log, normal_petri_net["structure"], normal_petri_net["initial_marking"], normal_petri_net["final_marking"], variant=replay_fitness.Variants.TOKEN_BASED)["log_fitness"]
			elif conformance_checking_technique == "alignment_based":
				aligned_traces = alignments.apply_log(test_log, normal_petri_net["structure"], normal_petri_net["initial_marking"], normal_petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_DIJKSTRA_NO_HEURISTICS)
				log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
			if(log_fitness >= thresholds[idx]):
				tn = tn + 1
				break
			if(idx == len(normal_petri_nets)-1):
				fp = fp + 1

	for id_log,anomalous_log in enumerate(anomalous_logs):
		for idx, normal_petri_net in enumerate(normal_petri_nets):
			parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
			if conformance_checking_technique == "token_based":
				log_fitness = replay_fitness.apply(anomalous_log, normal_petri_net["structure"], normal_petri_net["initial_marking"], normal_petri_net["final_marking"], variant=replay_fitness.Variants.TOKEN_BASED)["log_fitness"]
			elif conformance_checking_technique == "alignment_based":
				aligned_traces = alignments.apply_log(anomalous_log, normal_petri_net["structure"], normal_petri_net["initial_marking"], normal_petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_DIJKSTRA_NO_HEURISTICS)
				log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
			if(log_fitness >= thresholds[idx]):
				fn = fn + 1
				break
			if(idx == len(normal_petri_nets)-1):
				tp = tp + 1

	return tp, tn, fp, fn

def get_performance_metrics(tp, tn, fp, fn):

	try:
		accuracy = (tp+tn)/(tp+tn+fp+fn)
	except ZeroDivisionError:
		print("Accuracy could not be computed because the denominator was 0")
		accuracy = "undefined"

	try:
		precision = tp/(tp+fp)
	except ZeroDivisionError:
		print("Precision could not be computed because the denominator was 0")
		precision = "undefined"

	try:
		recall = tp/(tp+fn)
	except ZeroDivisionError:
		print("Recall could not be computed because the denominator was 0")
		recall = "undefined"
		
	try:
		f1 = 2*tp/(2*tp+fp+fn)
	except ZeroDivisionError:
		print("F1 could not be computed because the denominator was 0")
		f1 = "undefined"	

	diagnostics_file = open(output_testing_dir+"diagnostics.txt","w")
	diagnostics_file.write("TP="+str(tp)+"\nTN="+str(tn)+"\nFP="+str(fp)+"\nFN="+str(fn)+"\nAccuracy="+str(accuracy)+"\nPrecision="+str(precision)+"\nRecall="+str(recall)+"\nF1="+str(f1))
	diagnostics_file.close()


	return accuracy, precision, recall, f1

def write_elapsed_time(elapsed_time):
	file = open(output_testing_dir+"elapsed_time.txt", "w")
	file.write(str(elapsed_time) + " s\n")
	file.close()


try:
	debug_mode = int(sys.argv[1])
	conformance_checking_technique = sys.argv[2]
except IndexError:
	print("Not enough input arguments. Please, specify the debug mode")
	sys.exit()

anomalous_logs, test_logs = get_data()
patterns = get_patterns()
thresholds = get_thresholds()
if debug_mode == 1:
	elapsed_time = time.time()
tp,tn,fp,fn = classify_logs(anomalous_logs, test_logs, patterns, thresholds)
if debug_mode == 1:
	elapsed_time = time.time() - elapsed_time
	write_elapsed_time(elapsed_time)
accuracy, precision, recall, f1 = get_performance_metrics(tp, tn, fp, fn)


if debug_mode == 1:
	print("Classification results:")
	print("TP="+str(tp))
	print("TN="+str(tn))
	print("FP="+str(fp))
	print("FN="+str(fn))
	print("Accuracy: " + str(accuracy))
	print("Precision: " + str(precision))
	print("Recall: " + str(recall))
	print("F1: " + str(f1))
