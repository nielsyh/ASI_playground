from keras import optimizers, Sequential
from keras.layers import Input, Dense, concatenate, MaxPool2D, GlobalAveragePooling2D, Dropout, Conv2D, Flatten, LSTM
from keras.callbacks import Callback
from keras import optimizers
from metrics import Metrics
import keras
from keras.models import load_model
import calendar
import pvlib_playground
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint


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
    history = None
    day_month_to_predict = []

    def __init__(self, data, init_epochs, epochs, name, pred_csi = False):
        self.data = data
        self.init_train = True
        self.init_epochs = init_epochs
        self.epochs = epochs
        self.name = name
        self.pred_csi = pred_csi

    def set_days(self, days):
        self.day_month_to_predict = days

    def get_model(self):
        model = Sequential()
        print(self.data.train_x_df.shape)
        model.add(LSTM(50, activation='relu', input_shape=(self.data.train_x_df.shape[1], self.data.train_x_df.shape[2]), return_sequences=True))
        model.add(LSTM(25, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        opt = optimizers.Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt)
        self.model = model

    def set_model(self, nodes, activation, opt):
        model = keras.models.Sequential()
        model.add(
            LSTM(nodes[0], activation=activation, input_shape=(self.data.train_x_df.shape[1], self.data.train_x_df.shape[2]),
                 return_sequences=True))
        model.add(LSTM(nodes[1], activation=activation))
        model.add(Dense(nodes[2], activation=activation))
        model.add(Dense(nodes[1], activation=activation))
        model.add(Dense(1, activation=activation))
        model.compile(loss='mean_squared_error', optimizer=opt)
        self.model = model

    def train(self,epochs=50, batch_size=128):
        self.history = self.model.fit(self.data.train_x_df, self.data.train_y_df, epochs=epochs, batch_size=batch_size,
                                      validation_data=(self.data.val_x_df, self.data.val_y_df),
                                      callbacks=[TestCallback(self.data.test_x_df, self.data.test_y_df),
                                                 EarlyStopping(monitor='val_loss', patience=15),
                                                 ModelCheckpoint(filepath=str(self.name) + '.h5', monitor='val_loss', save_best_only=True)
                                                 ])
        return self.history


    def predict(self):
        self.model = load_model(str(self.name) + '.h5')
        y_pred =  self.model.predict(self.data.test_x_df)
        rmse, mae, mape = Metrics.get_error(self.data.test_y_df, y_pred)

        if self.pred_csi:
            # translate back to ghi
            pred_ghi  = []
            for idx, i in enumerate(y_pred):
                print('csi: ' + str(i))
                ghi  = pvlib_playground.PvLibPlayground.csi_to_ghi(i, int(self.data.test_x_df[idx][-1][1]),
                                                                   int(self.data.test_x_df[idx][-1][2]), int(self.data.test_x_df[idx][-1][3]),
                                                                   int(self.data.test_x_df[idx][-1][4]))
                print('ghi' + str(ghi))
                pred_ghi.append(ghi)
            return pred_ghi, rmse, mae, mape

        return y_pred, rmse, mae, mape

    def run_experiment(self):
        for exp in self.day_month_to_predict:
            print('LSTM SEQUENCE: ' + str(exp) + ', horizon: ' + str(self.data.pred_horizon))
            self.data.split_data_set(exp[0], exp[1])
            self.data.flatten_data_set_to_3d()
            self.get_model()

            epochs = self.epochs
            if self.init_train:
                epochs = self.init_epochs
                self.init_train = False
            self.train(epochs=epochs)

            y_pred, rmse, mae, mape = self.predict()
            Metrics.write_results_NN(str(self.name), self.data.test_x_df, self.data.test_y_df, y_pred, self.data.pred_horizon)

    def save_model(self):
        name = 'LSTM_' + str(self.data.month_split) + '_' + str(self.data.day_split) + '_' + str(self.data.pred_horizon)
        self.model.save(str(name) + '.h5')  # creates a HDF5 file 'my_model.h5'

    def load_model(self, path):
        self.model = load_model(str(path) + '.h5')