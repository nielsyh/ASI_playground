from keras.callbacks import Callback
from keras.regularizers import l2
from keras import optimizers
from metrics import Metrics
from keras.layers import Input, Dense, concatenate, MaxPool2D, GlobalAveragePooling2D, Dropout, Conv2D, Flatten
import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class TestCallback(Callback):
    def __init__(self, xtest, ytest):
        self.xtest = xtest
        self.ytest = ytest

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.xtest, self.ytest
        loss = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}\n'.format(loss))

class ANN_Multi():

    model = 0
    history = None
    day_month_to_predict = []

    def __init__(self, data, epochs, name):
        self.data = data
        self.epochs = epochs
        self.name = name

    def set_days(self, days):
        self.day_month_to_predict = days

    def set_model(self, nodes, activation, opt, drop_out):
        model = keras.models.Sequential()
        model.add(Dense(nodes[0], input_dim=(self.data.train_x_df.shape[1]), kernel_initializer='normal', activation='relu'))

        if drop_out > 0:
            model.add(Dropout(drop_out))

        model.add(Dense(nodes[1], activation=activation))

        if drop_out > 0:
            model.add(Dropout(drop_out))

        model.add(Dense(nodes[2], activation=activation))
        model.add(Dense(1, activation=activation))
        # opt = optimizers.Adam()
        model.compile(loss='mean_squared_error', optimizer=opt)
        self.model = model

    def get_model(self):
        model = keras.models.Sequential()
        model.add(Dense(124, input_dim=(self.data.train_x_df.shape[1]), kernel_initializer='normal', activation='relu'))
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Dense(124, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Dense(20, activation='linear'))
        opt = optimizers.Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt)
        self.model = model

    def train(self,epochs=50, batch_size=128):
        self.history = self.model.fit(self.data.train_x_df, self.data.train_y_df, epochs=epochs, batch_size=batch_size,
                                      validation_data=(self.data.val_x_df, self.data.val_y_df),
                                      callbacks=[TestCallback(self.data.test_x_df, self.data.test_y_df),
                                                 EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                                                 ModelCheckpoint(filepath= str(self.name) + '.h5', monitor='val_loss',
                                                                 save_best_only=True)
                                                 ])
        return self.history

    def predict(self):
        self.model = load_model(str(self.name) + '.h5')
        y_pred =  self.model.predict(self.data.test_x_df)
        rmse, mae, mape = Metrics.get_error(self.data.test_y_df, y_pred)
        return y_pred, rmse, mae, mape

    def run_experiment(self):
        for exp in self.day_month_to_predict:
            print('ANN SEQUENCE: ' + str(exp))
            self.data.split_data_set_EXPRMTL(exp[0], exp[1], 3)

            self.data.flatten_data_set_to_3d()
            self.data.test_x_df = self.data.test_x_df.reshape(self.data.test_x_df.shape[0], self.data.test_x_df.shape[1]*self.data.test_x_df.shape[2])
            self.data.train_x_df = self.data.train_x_df.reshape(self.data.train_x_df.shape[0], self.data.train_x_df.shape[1]*self.data.train_x_df.shape[2])
            self.data.val_x_df = self.data.val_x_df.reshape(self.data.val_x_df.shape[0], self.data.val_x_df.shape[1]*self.data.val_x_df.shape[2])

            self.get_model()

            epochs = self.epochs
            self.train(epochs=epochs)
            # self.plot_history()
            y_pred, rmse, mae, mape = self.predict()

            Metrics.write_results_multi(str(self.name), self.data.test_x_df.reshape(
                (self.data.test_x_df.shape[0],
                 self.data.sequence_len_minutes,
                 self.data.number_of_features)),
                                      self.data.test_y_df, y_pred)


    def save_model(self):
        name = 'ann_' + str(self.data.month_split) + '_' + str(self.data.day_split) + '_' + str(self.data.pred_horizon)
        self.model.save(str(name) + '.h5')  # creates a HDF5 file 'my_model.h5'

    def load_model(self, path):
        self.model = load_model(str(path) + '.h5')