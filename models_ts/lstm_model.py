import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.regularizers import l2
from keras import optimizers, Sequential

# from data_ts import *
from metrics import Metrics
from keras.layers import Input, Dense, concatenate, MaxPool2D, GlobalAveragePooling2D, Dropout, Conv2D, Flatten, LSTM
import keras
from keras.models import load_model
import calendar

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.regularizers import l2
from keras import optimizers

# from data_ts import *
from metrics import Metrics
from keras.layers import Input, Dense, concatenate, MaxPool2D, GlobalAveragePooling2D, Dropout, Conv2D, Flatten
import keras
from keras.models import load_model
import calendar


class TestCallback(Callback):
    def __init__(self, xtest, ytest):
        self.xtest = xtest
        self.ytest = ytest

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.xtest, self.ytest
        loss = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}\n'.format(loss))

class LSTM_predictor():

    model = 0

    def __init__(self, data, init_epochs, epochs, name):
        self.data = data
        self.init_train = True
        self.init_epochs = init_epochs
        self.epochs = epochs
        self.name = name

    def get_model(self):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(25, 18)))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1))
        opt = optimizers.Adam()
        model.compile(loss='mean_squared_error', optimizer=opt)
        self.model = model

    def train(self,epochs=50, batch_size=128):
        self.model.fit(self.data.train_x_df, self.data.train_y_df, epochs=epochs, batch_size=batch_size, validation_data=(self.data.val_x_df, self.data.val_y_df),
                       callbacks=[TestCallback(self.data.test_x_df, self.data.test_y_df)])

    def predict(self):
        y_pred =  self.model.predict(self.data.test_x_df)
        rmse, mae, mape = Metrics.get_error(self.data.test_y_df, y_pred)
        return y_pred, rmse, mae, mape

    def run_experiment(self):
        self.day_month_to_predict = []

        for m in self.data.months:
            last_day = calendar.monthrange(2019, m)[1]
            if m < 9:
                continue
            elif m == 9:
                days = list(range(11, last_day + 1)) #  Predict from 11 september
            else:
                days = list(range(1, last_day + 1))

            for d in days:
                self.day_month_to_predict.append((m, d))

        # self.day_month_to_predict = [(12,10), (12,20), (12,25)]
        # self.day_month_to_predict = [(9, 15)]

        for exp in self.day_month_to_predict:
            print('ANN SEQUENCE: ' + str(exp) + ', horizon: ' + str(self.data.pred_horizon))
            self.data.split_data_set(exp[0], exp[1])
            self.data.flatten_data_set_to_3d()
            self.get_model()

            epochs = self.epochs
            if self.init_train:
                epochs = self.init_epochs
                self.init_train = False

            self.train(epochs=epochs)
            y_pred, rmse, mae, mape = self.predict()

            name = 'LSTM_BETA_SEQUENCE'

            Metrics.write_results(str(self.name), self.data.test_x_df, self.data.test_y_df, y_pred, self.data.pred_horizon)

    def save_model(self):
        name = 'LSTM_' + str(self.data.month_split) + '_' + str(self.data.day_split) + '_' + str(self.data.pred_horizon)
        self.model.save(str(name) + '.h5')  # creates a HDF5 file 'my_model.h5'

    def load_model(self, path):
        self.model = load_model(str(path) + '.h5')