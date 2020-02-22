from keras.layers import Input, Dense, concatenate, MaxPool2D, GlobalAveragePooling2D, Dropout, Conv2D, Flatten
from keras.models import Model
import keras
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from tqdm import tqdm

from metrics import Metrics
import calendar
import numpy as np

class resnet50:
    # windowSize minimum 32 for resnet-50
    def __init__(self, data, init_epochs=200, epochs=50):
        self.data = data
        self.init_train = True
        self.init_epochs = init_epochs
        self.epochs = epochs

    def get_model(self, image_res):
        self.model = keras.models.Sequential()
        base = ResNet50(include_top=False, weights='imagenet',
                        input_shape=(image_res, image_res, 3))
        #freezeing all layers
        for layer in base.layers:
            layer.trainable = False
        self.model.add(base)
        # model.add(Conv2D(64, kernel_size=3, input_shape=(224,224,3)))
        self.model.add(Flatten())
        self.model.add(Dense(256, kernel_initializer='normal'))
        self.model.add(Dense(124))
        self.model.add(Dense(1))
        print(self.model.summary())
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, epochs=50, batch_size=128):
        self.model.fit(self.data.x_train, self.data.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.data.x_val, self.data.y_val))

    def predict(self):
        y_pred =  self.model.predict(self.data.x_test)
        rmse, mae, mape = Metrics.get_error(self.data.y_test, y_pred)
        return y_pred, rmse, mae, mape

    def save_model(self, name):
        for i in [2,3,4]:
            weights = self.model.layers[i].get_weights()
            np.save(str(name) + str(i), weights)
            # self.model.save(str(name) + '.h5')  # creates a HDF5 file 'my_model.h5'

    def load_model(self, name):
        # self.model = load_model(str(name) + '.h5')
        for i in [2,3,4]:
            weights  =np.load(str(name)+str(i)+'.npy')
            self.model.layers[i].set_weights(weights)

    def build_prem_models(self):
        prem = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]

        for tup in tqdm(prem, total=len(prem), unit='Days to predict'):
            self.data.split_data_set(tup[0], tup[1])
            self.data.flatten_data_set_CNN()

            self.get_model(400)

            epochs = self.epochs
            if self.init_train:
                epochs = self.init_epochs
                self.init_train = False

            self.train(epochs=epochs)
            name = str(tup[0]) + str(tup[1])
            self.save_model(name)
            print('Done: ' + str(name))

    def run_experiment(self):
        self.day_month_to_predict = []

        for m in self.data.months:
            last_day = calendar.monthrange(2019, m)[1]
            if m < 9:
                continue
            elif m == 9:
                days = list(range(11, last_day + 1))  # Predict from 11 september
            else:
                days = list(range(1, last_day + 1))

            for d in days:
                self.day_month_to_predict.append((m, d))

        for exp in self.day_month_to_predict:
            print('CNN: ' + str(exp) + ', horizon: ' + str(self.data.pred_horizon))
            self.data.split_data_set(exp[0], exp[1])
            self.data.flatten_data_set_CNN()

            epochs = self.epochs
            if self.init_train:
                epochs = self.init_epochs
                self.init_train = False

            self.train(epochs=epochs)
            y_pred, rmse, mae, mape = self.predict()

            name = 'CNN_BETA' + str(exp[0]) + str(exp[1])
            Metrics.write_results(str(name), self.data.x_test, self.data.y_test, y_pred, self.data.pred_horizon)
            self.save_model(name)
