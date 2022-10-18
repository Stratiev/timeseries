import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from generators.stochastic_processes import Brownian


SEED = 7
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    """
    Splits the timeseries into input batches (dataX)
    and a value to be predicted (dataY).
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# fix random seed for reproducibility
tf.random.set_seed(SEED)

path = Brownian()
path.generate_path(SEED)
df = pd.DataFrame({'Values': path.y, 'Time': path.t})
dataset = df.values
dataset = dataset.astype('float32')


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))


# reshape into X = t and Y = t + 1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# Reload model if it has already been trained.
model_name = 'models/saved_models/my_lstm'
try:
    model = tf.keras.models.load_model(model_name)
except OSError:
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
    model.save(model_name)


# make predictions
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# Prepare data for inversion
train_predict_reshaped = np.array([train_predict[:, 0], train[:-(look_back+1), 1]]).T
test_predict_reshaped = np.array([test_predict[:, 0], test[:-(look_back+1), 1]]).T

# invert predictions

train_predict_reshaped = scaler.inverse_transform(train_predict_reshaped)
test_predict_reshaped = scaler.inverse_transform(test_predict_reshaped)

# train_predict = scaler.inverse_transform(train_predict)
# trainY = scaler.inverse_transform([trainY])
# test_predict = scaler.inverse_transform(test_predict)
# testY = scaler.inverse_transform([testY])
# calculate root mean squared error
train_score = np.sqrt(mean_squared_error(trainY, train_predict[:, 0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = np.sqrt(mean_squared_error(testY, test_predict[:, 0]))
print('Test Score: %.2f RMSE' % (test_score))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(path.t, path.y)
ax1.plot(path.t[:6698], path.y[:6698], ls='-', label="Train data")
ax1.plot(train_predict_reshaped[:, 1], train_predict_reshaped[:, 0], ls='dotted', label="Train predictions")
ax1.plot(path.t[6702:], path.y[6702:], ls='-', label="Test data")
ax1.plot(test_predict_reshaped[:, 1], test_predict_reshaped[:, 0], ls='dotted', label="Test predictions")
error1 = [(x-y)/x for x, y in zip(path.y[:6698], train_predict_reshaped[:, 0])]
error2 = [(x-y)/x for x, y in zip(path.y[6702:], test_predict_reshaped[:, 0])]
error = error1 + error2
error_x_axis = path.t[:6698].tolist() + path.t[6702:].tolist()
ax2.plot(error_x_axis[500:], error[500:], ls='dotted', alpha=0.4, label="Relative Error")
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
