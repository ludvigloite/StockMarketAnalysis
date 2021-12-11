import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from support_functions import *
from tensorflow import keras

# configure
n_seq = 30
n_test = 200
n_lag = 120
n_epochs = 20
n_batch = 1
n_neurons = 120

#index to the test case you want to plot
test_index = 100

#Configure dataset
PATH_TO_STOCKS = "../../Datasets/Stocks/"
STOCK = "fl.us.txt"
SELECTED = "Close"

#Choose if you want to use a pretrained model to predict and plot or train a new one
TRAIN_NEW_MODEL = False
MODEL_PATH = './LSTM_model_30days'

if __name__ == "__main__":
    # load dataset
    data = pd.read_csv(os.path.join(PATH_TO_STOCKS, STOCK), delimiter = ",", parse_dates=['Date'], usecols=[0,1,2,3,4], index_col=['Date'])
    series = data[SELECTED]
    scaler, X_train, y_train, X_test, y_test = prepare_data(series, n_test, n_lag, n_seq)
    
    if TRAIN_NEW_MODEL or not os.path.isdir(MODEL_PATH) :
        model = create_model(X_train, y_train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

        #Have to rehsape the training data to fit the model [samples, 1, features]
        X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

        #Fit the model with the training data
        es = tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss')
        for i in range(n_epochs):
            history = model.fit(X_train_reshaped, y_train, epochs=1, verbose=1, batch_size=n_batch, shuffle=False, callbacks=[es])
            model.reset_states()

        # save model
        model.save(MODEL_PATH)
    
    else:
        model = keras.models.load_model(MODEL_PATH)
        
    # make predictions
    preds = make_predictions(model, n_batch, X_test, n_lag, n_seq)
    preds_diff = preds
    
    # inverse transform forecasts and test
    preds = inverse_transform(series, preds, scaler, n_test+n_seq-1)
    actual = [row.reshape(-1) for row in y_test]
    actual = inverse_transform(series, actual, scaler, n_test+n_seq-1)
    
    # evaluate predictions
    evaluate_predictions(actual, preds, n_lag, n_seq)

    # plot forecasts
    plot_predictions(series, preds, n_test+n_seq-1, test_index, MODEL_PATH.split('/')[-1])