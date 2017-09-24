# Model and functions to train the simulator

import pandas as pd #data analysis
import numpy as np #matrix op
from sklearn.model_selection import train_test_split #split train, val, test data
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from imgTools import INPUT_SHAPE, batch_generator
import argparse # command line arguments
import os # read files 

# fro debugging, seed(0) allows reproducible results
np.random.seed(0)

def load_data(args):
	"""
	Load training data and split training and validation set
	"""
	data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'))#, 
			#names=['center','left','right','steering','throttle','reverse','speed'])
	

	# inputs (images)
	x = data_df[['center','left','right']].values
	# labels (steering angles)
	y = data_df['steering'].values


	x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=args.test_size, random_state=0)

	return x_train, x_validation, y_train, y_validation


def build_model(args):
	"""
	Model used from End to End Driving paper from NVIDIA
	Image normalization to avoid saturation
	Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
	Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
	Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
	Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
	Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
	Dropout 0.5
	Fully connected: neurons: 100, activation: ELU
	Fully connected: neurons: 50, activation: ELU
	Fully connected: neurons: 10, activation: ELU
	Fully connected: neurons: 1, activation: Linear
	
	ELU (Exponential Linear Unit) to address the vanishing gradient
	""" 

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=INPUT_SHAPE))
	model.add(Conv2D(24, 5, 5, activation='elu',subsample=(2,2)))
	model.add(Conv2D(36, 5, 5, activation='elu',subsample=(2,2)))
	model.add(Conv2D(48, 5, 5, activation='elu',subsample=(2,2)))
	model.add(Conv2D(64, 3, 3, activation='elu'))
	model.add(Conv2D(64, 3, 3, activation='elu'))
	model.add(Dropout(args.keep_prob))
	model.add(Flatten())
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))
	model.summary()

	return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = load_data(args)
    #build model
    model = build_model(args)
    #train model on data, it saves as model.h5 
    train_model(model, args, *data)


if __name__ == '__main__':
    main()





