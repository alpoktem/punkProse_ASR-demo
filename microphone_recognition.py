import _thread
import pyaudio
import wave
import speech_recognition as sr
import os
from shutil import copyfile
import sys
from proscript.proscript import Word, Segment, Proscript
from proscript.utilities import utils
from credentials import GOOGLE_CLOUD_SPEECH_CREDENTIALS

WORKING_DIR = 'rec'
FILESAVE_PREFIX = 'recorded'
SPEAKER_ID = 'spk1'

MAX_SEGMENT_LENGTH = 30.0 #SECONDS
MFA_ALIGN_BINARY = "/Users/alp/extSW/montreal-forced-aligner/bin/mfa_align"
MFA_LEXICON = "/Users/alp/extSW/montreal-forced-aligner/pretrained_models/en.dict"
MFA_LM = "/Users/alp/extSW/montreal-forced-aligner/pretrained_models/english.zip"

#WAV properties
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

#other parameters
SEGMENT_END_BUFFER = 0.15

def input_thread(a_list):
	input()
	a_list.append(True)

def record_audio(WAVE_OUTPUT_FILENAME, FORMAT, CHANNELS, RATE, CHUNK, raw_output=False):
	a_list = []
	_thread.start_new_thread(input_thread, (a_list,))

	# start Recording
	audio = pyaudio.PyAudio()
	stream = audio.open(format=FORMAT, channels=CHANNELS, 
						rate=RATE, input=True, frames_per_buffer=CHUNK)
	print("recording...")
	frames = []

	while not a_list:
		data = stream.read(CHUNK)
		frames.append(data)
	print("finished recording")
	# stop Recording
	stream.stop_stream()
	stream.close()
	audio.terminate()

	if raw_output == False:
		waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		waveFile.setnchannels(CHANNELS)
		waveFile.setsampwidth(audio.get_sample_size(FORMAT))
		waveFile.setframerate(RATE)
		waveFile.writeframes(b''.join(frames))
		waveFile.close()
	else:
		rawbytes = b''.join(frames)
		with open(RAW_OUTPUT_FILENAME, 'wb') as f:
			f.write(rawbytes)

def determine_recording_id(working_dir_name, wav_in=None):
	if wav_in and os.path.dirname(os.path.abspath(wav_in)) == os.path.abspath(working_dir_name):
		return os.path.splitext(os.path.basename(wav_in))[0]

	last_id = -1
	for file in os.listdir(working_dir_name):
		if file.endswith(".wav") and FILESAVE_PREFIX in file:
			curr_id = int(file.split('.')[0].split('_')[1])
			if curr_id > last_id:
				last_id = curr_id
	return FILESAVE_PREFIX + '_' + str(last_id+1)

def run_microphone_recognizer(working_dir_name, recognizer, wav_in = None, csv_in = None):
	#-----------TAKE THIS PART OUT ON A DIFFERENT FUNCTION------------
	if not os.path.exists(working_dir_name):
		os.makedirs(working_dir_name)

	RECORDING_ID = determine_recording_id(WORKING_DIR, wav_in)
	print("Recording ID: " + RECORDING_ID)

	WAVE_OUTPUT_FILENAME = RECORDING_ID + '.wav'
	CSV_OUTPUT_FILENAME = RECORDING_ID + '.0.csv'  #proscript without punctuation
	TXT_OUTPUT_FILENAME = RECORDING_ID + '.txt'

	txt_out = os.path.join(working_dir_name, TXT_OUTPUT_FILENAME)
	csv_out = os.path.join(working_dir_name, CSV_OUTPUT_FILENAME)

	#record from microphone or read the pre-recorded file
	if wav_in == None:
		wav_out = os.path.join(working_dir_name, WAVE_OUTPUT_FILENAME)
		frames = record_audio(wav_out, FORMAT, CHANNELS, RATE, CHUNK)
	else:
		wav_out = os.path.join(working_dir_name, WAVE_OUTPUT_FILENAME)
		if not os.path.dirname(wav_in) == working_dir_name:
			copyfile(wav_in, wav_out)

		#recognition is already done and written in csv
		if not csv_in == None:
			csv_out = os.path.join(working_dir_name, CSV_OUTPUT_FILENAME)
			if not os.path.dirname(csv_in) == working_dir_name:
				copyfile(csv_in, csv_out)
			return csv_out, None
	#-----------TAKE THIS PART OUT ON A DIFFERENT FUNCTION------------

	#recognize with google cloud speech API
	print("Sending to Google Cloud speech API")
	with sr.AudioFile(wav_out) as source:
	     audio = recognizer.record(source)  # read the entire audio file
	try:
		response = recognizer.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS, show_all=True)
		print("Google Cloud response:\n")
		#print(response)
		#response = {'results': [{'alternatives': [{'transcript': "this is a warning I'm giving to you I'm kind of crazy", 'confidence': 0.9657507, 'words': [{'startTime': '0.100s', 'endTime': '0.500s', 'word': 'this'}, {'startTime': '0.500s', 'endTime': '0.600s', 'word': 'is'}, {'startTime': '0.600s', 'endTime': '0.600s', 'word': 'a'}, {'startTime': '0.600s', 'endTime': '0.900s', 'word': 'warning'}, {'startTime': '0.900s', 'endTime': '1.100s', 'word': "I'm"}, {'startTime': '1.100s', 'endTime': '1.400s', 'word': 'giving'}, {'startTime': '1.400s', 'endTime': '1.600s', 'word': 'to'}, {'startTime': '1.600s', 'endTime': '1.700s', 'word': 'you'}, {'startTime': '1.700s', 'endTime': '2.500s', 'word': "I'm"}, {'startTime': '2.500s', 'endTime': '2.800s', 'word': 'kind'}, {'startTime': '2.800s', 'endTime': '2.800s', 'word': 'of'}, {'startTime': '2.800s', 'endTime': '3.300s', 'word': 'crazy'}]}]}]}
		
		#print("Google Cloud could not understand audio")
	except sr.RequestError as e:
		print("Could not request results from Google Cloud service; {0}".format(e))

	if response:
		#represent information in proscript format
		duration = len(audio.frame_data) / audio.sample_rate / audio.sample_width
		print('duration', duration)
		
		p = Proscript()
		p.audio_file = wav_out
		p.speaker_ids = [SPEAKER_ID]
		p.id = RECORDING_ID
		p.duration = duration

		complete_transcription = ""
		for segment_no, recognized_segment in enumerate(response['results']):
			transcription = recognized_segment['alternatives'][0]['transcript']
			confidence = recognized_segment['alternatives'][0]['confidence']
			wordData = recognized_segment['alternatives'][0]['words']
			s = Segment()
			s.transcript = transcription
			s.speaker_id = SPEAKER_ID
			s.id = segment_no + 1
			s.start_time = float(wordData[0]['startTime'][:-1])
			s.end_time = float(wordData[-1]['endTime'][:-1]) + SEGMENT_END_BUFFER
			p.add_segment(s)
			complete_transcription += transcription + " "

		print("Google Cloud recognized: %s"%complete_transcription)

		utils.proscript_segments_to_textgrid(p, WORKING_DIR, p.id, speaker_segmented=False)
		try:
			utils.mfa_word_align(WORKING_DIR, mfa_align_binary=MFA_ALIGN_BINARY, lexicon=MFA_LEXICON, language_model=MFA_LM)
			mfa_failed = False
		except:
			mfa_failed = True

		if not mfa_failed:
			utils.get_word_features_from_textgrid(p, prosody_tag=True, remove_textgrid=True)
			utils.assign_word_ids(p)
			utils.assign_pos_tags(p.get_last_segment())

			p.get_speaker_means()
			utils.assign_acoustic_means(p)

			#write transcription to text file
			txt_out = os.path.join(working_dir_name, TXT_OUTPUT_FILENAME)
			with open(txt_out, 'w') as f:
				f.write(complete_transcription)

			#write proscript to csv
			p.to_csv(csv_out)
			return csv_out, complete_transcription
		else: 
			print("MFA failed")
			#Remove wav and textgrid from directory
			return None, None
	else:
		return None, None

if __name__ == '__main__':
	#load recognition tools
	r = sr.Recognizer()

	if len(sys.argv) > 1:
		wav_in = sys.argv[1]
		proscript_path, transcription = run_microphone_recognizer(WORKING_DIR, r, wav_in)
	else: 
		proscript_path, transcription = run_microphone_recognizer(WORKING_DIR, r)
	
	print("Proscript written to: %s"%proscript_path)