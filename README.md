# punkProse_ASR-demo
This is a demo software that contains scripts to punctuate audio recordings using punkProse library. It is inteded to use for demonstration purposes. 

## Installation

* Requirements: 
	- Python 3.x
	- Numpy
	- Theano
	- yaml 
  - [Proscript](https://github.com/alpoktem/proscript)
  - [Montreal forced aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)
  - [Praat](http://www.fon.hum.uva.nl/praat/)
  
* Setup:
  Install the required python packages. Install Montreal forced aligner and link the files in `microphone_recognition.py`. Praat should be installed and accessible from command line as `praat`
  Currently two English punctuation models are provided under the directory `models`. 
  
## Run
In order to run type:
`python listen_and_punctuate.py`

You can choose either to record from microphone or open a pre-recorded audio file. Raw and punctuated proscripts will be created under the directory `rec`. 

## Visualizing output on Prosograph
In order to visualize the recordings install [Prosograph](https://github.com/alpoktem/Prosograph) and link the directory `rec` in file `dataconfig_newdata.py` under Prosograph. You can switch between different punctuation outputs using number keys. 

## Sample demo setup
![Demo setup with Prosograph](https://raw.githubusercontent.com/alpoktem/punkProse_ASR-demo/master/images/interface-5.png)

## Read more
This demo was presented in Interspeech 2018:

	@inproceedings{punkProse,
		author = {Alp Oktem and Mireia Farrus and Antonio Bonafonte},
		title = {Visualizing Punctuation Restoration in Speech Transcripts with Prosograph},
		booktitle = {Interspeech 2018},
		year = {2018},
		address = {Hyderabad, India}
	}
