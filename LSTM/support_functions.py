from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

 
#Function to create the input data and the corresponding labels to be predicted
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
        raw_values = series.values
        
        # transform to stationary data
        diff_series = series.diff().dropna()
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        
        # rescale values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        
        # transform into supervised learning problem X, y
        X, y = split_sequence(scaled_values, n_lag, n_seq)
        X_train, y_train = X[:-n_test], y[:-n_test]
        X_test, y_test = X[-n_test:], y[-n_test:]
        return scaler, X_train, y_train, X_test, y_test
    
    

# Create a simple LSTM model
def create_model(X_train, y_train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        
        # Create model topology
        model = Sequential()
        model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X_train.shape[1], X_train.shape[2]), stateful=True, return_sequences=False))
        model.add(Dense(units = y_train.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanAbsoluteError()])
        return model

# Make a single prediction
def predict_lstm(model, X, n_batch):
	X = X.reshape(1, 1, len(X))
	
    # make prediction
	pred = model.predict(X, batch_size=n_batch)
    
	return [x for x in pred[0, :]]

#Function that loops through the test data and gets the predictions
def make_predictions(model, n_batch, X_test, n_lag, n_seq):
	preds = list()
	for i in range(X_test.shape[0]):
		pred = predict_lstm(model, X_test[i], n_batch)
		preds.append(pred)
	return preds

# Undo the differencing done to make data stationary
def inverse_difference(last_ob, pred):
    pred_cumsum = pred.cumsum()
    inverted = [last_ob]*len(pred)
    inverted = [sum(x) for x in zip(inverted, pred_cumsum)]
    return inverted

# Inversly transforms the data back to proper scale
def inverse_transform(series, preds, scaler, n_test):
    inverted = list()
    for i in range(len(preds)):
        # create array from pred
        pred = np.array(preds[i])
        pred = np.array(preds[i]).reshape(1, len(pred))

        # invert scaling
        inv_scale = scaler.inverse_transform(pred)
        inv_scale = inv_scale[0, :]

        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted

def evaluate_predictions(test, preds, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [pred[i] for pred in preds]
		rmse = np.sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
        

# plot the predictions in the context of the original dataset
def plot_predictions(series, preds, n_test, test_index):
    assert test_index<len(preds) and test_index>=0, \
        f'Please input a valid test_index. Must be larger than 0 and less than {len(preds)}'
    
    x_lim = int(len(series) * 0.95)
    y_lim_upper = np.max(preds) * 1.01
    y_lim_lower = min(np.min(series[x_lim:]), np.min(preds))*0.99
    
    # plot the entire dataset in blue
    plt.figure(figsize=(16,8))
    plt.plot(series.values, '-o', label='Actual')
    
    # plot the prediction
    start_index = len(series) - n_test + test_index - 1
    end_index = start_index + len(preds[test_index]) + 1
    xaxis = [x for x in range(start_index, end_index)]
    yaxis = [series.values[start_index]] + preds[test_index]
    plt.plot(xaxis, yaxis, '-o', color='orange', label='Predicted')
        
    # show the plot
    plt.axis([x_lim, len(series)+len(preds[0]), y_lim_lower, y_lim_upper])
    plt.legend()
    plt.savefig("lstm_plot.png")
    plt.show()
    return