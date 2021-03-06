'''
Training only classification network on auxillary tasks using the features from the DL model trained on primary task
'''
import sys
sys.path.append("..")

from train_classifier_model import classification_params
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import csv, numpy as np
from example_training_loop import classification_net_features
import tensorflow as tf


########################################################################################
# Hard-coded rules. Please change this if you are changing the dataset format
########################################################################################
# 1. Provide the order in which the labels are provided in the label matrix
task_mapping = {"shape": 0, "color": 1, "size": 2, "quadrant": 3, "background": 4}


def classifier_network(data_shape, num_labels):
	""" Build a small fully connected two hiddey layer neural network classifier. Can replace this with any custom classifier of choice"""
	
	model = Sequential()
	model.add(Dense(512, input_shape=(data_shape, ), activation='relu', name="cus_dense_1"))
	model.add(Dropout(0.5, name='cus_dense_do_1'))
	model.add(Dense(100, activation='relu', name='cus_dense_2'))
	model.add(Dropout(0.3, name='cus_dense_do_2'))
	model.add(Dense(num_labels, activation='softmax', name='cus_dense_3'))
	return model

def train_models():
	""" Train the features extracted from a DL model to perform auxillary tasks"""
	
	trained_task = classification_params.task
	header_row = ["task"] + list(task_mapping.keys())
	fd = open(classification_params.OUTPUT_MATRIX_FILE_NAME,'w')
	writer = csv.writer(fd)
	writer.writerow(header_row)

	# shape - 0, color - 1, size - 2, quadrant - 3, background - 4
	for task_key in task_mapping.keys():
		if(trained_task == 'all' or trained_task == task_key ):
			print("==================================================")
			print("Using features of DL model original trained for: ", task_key)

			print("Loading training data features ...... ")
			train_data = np.load(classification_params.train_features_path)
			
			print("Loading test data features ...... ")
			test_data = np.load(classification_params.test_features_path)

			accuracy_list = []
			for predict_task_key, predict_task_value in task_mapping.items():
				#lable_index = predict_task_value
				print("Predicting the task for: ", predict_task_key)

				# Processing training data
				train_x = train_data["data"]
				train_y = train_data["lables"][:,predict_task_value]
				num_labels = len(set(train_y))
				print("Traning data shape: ", train_x.shape)
				print("Traning label shape: ", train_y.shape)
				print("Number of actual labels: ", num_labels)
				print("Train lables set: ", set(train_y))

				print("Training data pre-processing and shuffling...")
				total_data_point = train_x.shape[0]
				data_indices = np.arange(total_data_point)
				np.random.shuffle(data_indices)
				train_x = train_x.take(data_indices, axis=0)
				train_y = train_y.take(data_indices, axis=0)
				data_shape = train_x.shape[1]
				train_y_one_hot = to_categorical(train_y, num_labels)
				train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
				model = classifier_network(data_shape, num_labels)
				print(model.summary())
				#tf.enable_eager_execution()
				classification_net_features(train_dataset=train_dataset,model=model,train_x_list=train_x)



				print("Training the classifier ...... ")
				adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
				model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
				model.fit(train_x, train_y_one_hot, batch_size=classification_params.batch_size, epochs=classification_params.epoch, verbose=classification_params.verbose, validation_split=classification_params.validation_split)

				# Processing test data
				test_data_x = test_data["data"]
				test_data_answer = test_data["lables"][:,predict_task_value]
				num_labels = len(set(test_data_answer))
				print("Test data shape: ", test_data_x.shape)
				print("Test labels shapes: ", test_data_answer.shape)
				print("Test labels set: ", set(test_data_answer))
				test_data_answer_one_hot = to_categorical(test_data_answer, num_labels)
				score = model.evaluate(x=test_data_x, y=test_data_answer_one_hot, verbose=2)
				print("\n######## Note this thing ########\nPerformance on features original trained to predict",
						task_key, "and later used to predict", predict_task_key, "is: ", score, "\n")

				accuracy_list.append(round(score[1],4))

			writer.writerow([task_key] + accuracy_list)

if __name__=="__main__":
	train_models()
