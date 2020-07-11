# cnn-bodylandmarking-transfer-learning

Python libraries used

	python version 3.7
	pytorch 1.3.1
	tensorboardX (tensorflow not required)
	CUDA 10.1
	scikit-image 
	torchvision
	
detailed training and validation log:

	go to terminal and 'cd' to the project directory
	enter 'tensorboard --logdir runs/'
	open 'localhost:6006' in web browser
	
clean and compiled custlr dataset can be obtained here: 

https://drive.google.com/file/d/1Pv35fecmvEbC4x8NjgwDDJAmXR1VobQL/view?usp=sharing 

extract the zip and insert the 'custlr' folder in 'dataset' folder

# Directory

DATASET

dataset/

	custlr/ (download from google drive: https://drive.google.com/file/d/1Pv35fecmvEbC4x8NjgwDDJAmXR1VobQL/view?usp=sharing)
	auged/: stores the original and augmented dataset and information to compile expanded dataset landmarks
		custlrinfo101.csv: loader for the original 101 custlr dataset
		train.csv: loader for the augmented training set
		val.csv: loader for the augmented validation set
		test.csv: loader for the augmented test set
		train2x.csv: loader for the augmented training set with alternate landmarks
		val2x.csv: loader for the augmented validation set with alternate landmarks
		test2x.csv: loader for the augmented test set with alternate landmarks
		train13lm.csv: loader for the augmented training set samples with all 13 landmarks
		test13lm.csv: loader for the augmented test set samples with all 13 landmarks

OUTPUT AND VISUALIZATION

models/

	custlrexpanded_models/: stores model results from dataset expansion experiment
	iterateparam_models/: stores model results from parameter iteration experiment
	lmfulltrain.pkl: contains model weights from training with Deepfashion from https://github.com/fdjingyuan/Deep-Fashion-Analysis-ECCV2018

pred_visualized/

	custlrexpanded_plot/: stores prediction plotting of dataset expansion experiment
	iterateparam_plot/:stores prediction plotting of parameter iteration experiment

runs/: stores tensorboard log

FROM SOURCE MODEL

src/

	base_network.py: base network of vgg16
	lm_network.py: stores the default network model and requirement variables
	const: contains variables initialized for dataset and model
	utils: contains function for calculating validation loss
	conf/lm.py: contains variables for landmarking task


CLASSES

	custlrloader.py: 
	class for loading dataset and functions for preprocessing

	modelclass.py: 
	class that generates the neural network model used throughout experiment

FUNCTIONS

	transfertraining_function.py: 
	function that loads the model with deepfashion weights to run transfer learning with custlr dataset

	prediction_function.py: 
	function for running prediction of a model

	custlrexpanded_compare_pred_function.py: 
	function that runs prediction and compares the result of augmented dataset and augmented dataset with landmarks expansion

RUNNABLE FILES

	iterateparam_prediction_run.py: 
	script for running prediction of parameter iteration

	iterateparam_training_run.py: 
	script to run training for parameter iteration experiment

	custlrexpanded_experiment_run.py: 
	script to run training using augmented dataset and expanded dataset and run prediction comparison

	predict_13landmarks_run.py
	script to run the expanded dataset model prediction against all 13 landmarks
