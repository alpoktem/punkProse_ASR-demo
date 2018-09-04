import microphone_recognition as mr
import speech_recognition as sr
import sys
sys.path.insert(0, "punkProse")
import models
import punctuator
import utilities
import theano
import os
import yaml
import copy
sys.path.insert(0, "Proscript/proscript")
from proscript.proscript import Word, Segment, Proscript
from proscript.utilities import utils

model_file_w = "models/punkProse_model_eng_wordonly.pcl"
model_file_wPOSpmf = "models/punkProse_model_eng_prosodic.pcl"
config_file = "punkProse/parameters.yaml"
working_dir_name = mr.WORKING_DIR

if __name__ == '__main__':
	print("Loading configurations file...")
	with open(config_file, 'r') as ymlfile:
		config = yaml.load(ymlfile)

	print("Loading model parameters...")
	net_wPOSpmf, inputs_wPOSpmf, input_feature_names_wPOSpmf, _ = models.load(model_file_wPOSpmf, 1)
	net_w, inputs_w, input_feature_names_w, _ = models.load(model_file_w, 1)
	#net_wp, inputs_wp, input_feature_names_wp, _ = models.load(model_file_wp, 1)
	print("Building model...")
	predict_wPOSpmf = theano.function(inputs=inputs_wPOSpmf, outputs=net_wPOSpmf.y)
	predict_w = theano.function(inputs=inputs_w, outputs=net_w.y)
	#predict_wp = theano.function(inputs=inputs_wp, outputs=net_wp.y)
	print("Loading vocabularies...")
	#vocabulary_dict, leveler_dict = punctuator.load_dictionaries(config, input_feature_names_w)
	vocabulary_dict, leveler_dict = punctuator.load_dictionaries(config, input_feature_names_wPOSpmf)

	#load recognition tools
	r = sr.Recognizer()

	while True:
		key_input = input("Press R to record, O to open audio file, P to open proscript file, Q to quit...")
		print(key_input)
		if key_input == "q" or key_input=="Q":
			sys.exit()
		elif key_input == "r" or key_input=="R" or key_input=="":
			#run recognition
			proscript_path, transcription = mr.run_microphone_recognizer(working_dir_name, r)
		elif key_input == "o" or key_input=="O":
			file_input = input("Input filename: ")
			file_input = file_input.strip()
			if os.path.isfile(file_input) and file_input.endswith(".wav"):
				proscript_path, transcription = mr.run_microphone_recognizer(working_dir_name, r, wav_in = file_input.strip())
			else:
				print("File doesn't exist")
				proscript_path = None
		elif key_input == "p" or key_input=="P":
			csv_file_input = input("Input filename: ")
			csv_file_input = csv_file_input.strip()
			if os.path.isfile(csv_file_input) and csv_file_input.endswith(".csv"):
				wav_file_input = os.path.join(os.path.dirname(csv_file_input), os.path.splitext(os.path.basename(csv_file_input))[0] + '.wav')
				print(wav_file_input)
				if os.path.isfile(wav_file_input):
					proscript_path, transcription = mr.run_microphone_recognizer(working_dir_name, r, wav_in=wav_file_input, csv_in=csv_file_input) #only used for copying the audio to working dir
					transcription = None
				else:
					print("Proscript is not accompanied with its audio")
			else:
				print("File doesn't exist")
				proscript_path = None
		else:
			proscript_path = None

		#punctuate the input
		if proscript_path:
			#punctuate proscript
			proscript_data = utilities.read_proscript(proscript_path, add_end=True)
			if transcription == None:
				transcription = ' '.join(proscript_data['word'])

			punctuated_proscript_data_w = copy.copy(proscript_data)
			punctuated_transcript_w = punctuator.restore_unsequenced_test_data( punctuated_proscript_data_w,
												  	   vocabulary_dict=vocabulary_dict,
												  	   leveler_dict=leveler_dict,
												  	   predict_function=predict_w, 
												  	   input_feature_names=input_feature_names_w, 
												  	   sequence_length=config["SAMPLE_SIZE"],
												  	   readable_format=True)

			# punctuated_proscript_data_wp = copy.copy(proscript_data)
			# punctuated_transcript_wp = punctuator.restore_unsequenced_test_data( punctuated_proscript_data_wp,
			# 									  	   vocabulary_dict=vocabulary_dict,
			# 									  	   leveler_dict=leveler_dict,
			# 									  	   predict_function=predict_wp, 
			# 									  	   input_feature_names=input_feature_names_w, 
			# 									  	   sequence_length=config["SAMPLE_SIZE"],
			# 									  	   readable_format=True)


			punctuated_proscript_data_wPOSpmf = copy.copy(proscript_data)
			punctuated_transcript_wPOSpmf = punctuator.restore_unsequenced_test_data( punctuated_proscript_data_wPOSpmf,
													  	   vocabulary_dict=vocabulary_dict,
													  	   leveler_dict=leveler_dict,
													  	   predict_function=predict_wPOSpmf, 
													  	   input_feature_names=input_feature_names_wPOSpmf, 
													  	   sequence_length=config["SAMPLE_SIZE"],
													  	   readable_format=True)
			



			#Write returned proscripts to file in the input dir.
			#working_dir = os.path.dirname(proscript_path)
			recording_id = os.path.basename(proscript_path).split('.')[0]

			#print out stuff
			print("################################################################")
			print("################# UNPUNCTUATED TRANSCRIPTION ###################")
			print(transcription)
			unpunctuated_proscript_id = recording_id + '.0'
			unpunctuated_proscript_file = os.path.join(working_dir_name, unpunctuated_proscript_id + '.csv')
			proscript_data['punctuation_after'] = [''] * len(proscript_data['punctuation_after'])
			p0 = Proscript()
			p0.from_dict(proscript_data, unpunctuated_proscript_id)
			p0.to_csv(unpunctuated_proscript_file)
			print("################# PUNCTUATED WITH punkProse ####################")
			print("Model 1: word")
			print(punctuated_transcript_w)
			punctuated_proscript_id = recording_id + '.1'
			punctuated_proscript_file = os.path.join(working_dir_name, punctuated_proscript_id + '.csv')
			p1 = Proscript()
			p1.from_dict(punctuated_proscript_data_w, punctuated_proscript_id)
			p1.to_csv(punctuated_proscript_file)
			print("----------------------------------------------------------------")
			print("----------------------------------------------------------------")
			# print("Model 2: word+pause")
			# print(punctuated_transcript_wp)
			# punctuated_proscript_id = recording_id + '.2'
			# punctuated_proscript_file = os.path.join(working_dir_name, punctuated_proscript_id + '.csv')
			# p2 = Proscript()
			# p2.from_dict(punctuated_proscript_data_wp, punctuated_proscript_id)
			# p2.to_csv(punctuated_proscript_file)
			# print("----------------------------------------------------------------")
			# print("----------------------------------------------------------------")
			print("Model 2: word + POS + pause + f0_mean")
			print(punctuated_transcript_wPOSpmf)
			print("################################################################")
			punctuated_proscript_id = recording_id + '.2'
			punctuated_proscript_file = os.path.join(working_dir_name, punctuated_proscript_id + '.csv')
			p3 = Proscript()
			p3.from_dict(punctuated_proscript_data_wPOSpmf, punctuated_proscript_id)
			p3.to_csv(punctuated_proscript_file)
	
		else:
			print("Can't recognize")