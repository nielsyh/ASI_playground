import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback

from data import Data
from metrics import Metrics
# from models.model_template import Predictor_template
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

class ANN_Predictor():

    model = 0

    def __init__(self, data,init_epochs=200, epochs=50):
        self.data = data
        self.init_train = True
        self.init_epochs = init_epochs
        self.epochs = 50

    def get_model(self):
        model = keras.models.Sequential()
        model.add(Dense(256, input_dim=(31), kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model

    def train(self,epochs=50, batch_size=128):
        self.model.fit(self.data.x_train, self.data.y_train, epochs=epochs, batch_size=batch_size,
                       callbacks=[TestCallback(self.data.x_test, self.data.y_test)])

    def predict(self):
        y_pred =  self.model.predict(self.data.x_test)
        rmse, mae, mape = Metrics.get_error(self.data.y_test, y_pred)
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

        self.day_month_to_predict = [(12,10), (12,20), (12,30)]
        self.day_month_to_predict = [(7, 28)]

        for exp in self.day_month_to_predict:
            print('ANN: ' + str(exp) + ', horizon: ' + str(self.data.pred_horizon))
            self.data.split_data_set(exp[0], exp[1])
            self.data.flatten_data_set()
            self.data.normalize_data_sets()

            epochs = self.epochs
            if self.init_train:
                epochs = self.init_epochs
                self.init_train = False

            self.train(epochs=epochs)
            y_pred, rmse, mae, mape = self.predict()

            name = 'ANN_BETA'

            name = name + '_horizon_' + str(self.data.pred_horizon)
            if self.data.debug:
                name = name + '_debug'
            if self.data.images:
                name = name + '_images'
            if self.data.meteor_data:
                name = name + '_meteor'

            Metrics.write_results(str(name), self.data.x_test, self.data.y_test, y_pred, self.data.pred_horizon)

    def save_model(self):
        name = 'ann_' + str(self.data.month_split) + '_' + str(self.data.day_split) + '_' + str(self.data.pred_horizon)
        self.model.save(str(name) + '.h5')  # creates a HDF5 file 'my_model.h5'

    def load_model(self, path):
        self.model = load_model(str(path) + '.h5')
