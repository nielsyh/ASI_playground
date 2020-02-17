from sklearn.svm import SVC
from dataframe_sequence import DataFrameSequence
from metrics import Metrics
import pickle
import calendar
from sklearn.model_selection import GridSearchCV
import sys


class SVM_predictor:
    day_month_to_predict = []

    def __init__(self, data, name):
        self.data = data
        self.model = 0
        self.name = name

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

        prem = [(10,5), (10,6), (10,7), (10,8), (10,20)]
        self.day_month_to_predict = prem

        for exp in self.day_month_to_predict:
            sys.stdout.write('SVM: ' + str(exp) + ', horizon: ' + str(self.data.pred_horizon))
            self.data.split_data_set(exp[0], exp[1])
            self.data.flatten_data_set()
            self.train()
            y_pred, rmse, mae, mape = self.predict()

            Metrics.write_results_SVR(str(self.name), self.data.test_x_df.reshape(
                (self.data.test_x_df.shape[0],
                 self.data.sequence_len_minutes,
                 self.data.number_of_features)),
                                      self.data.test_y_df, y_pred,
                                      self.data.pred_horizon)

    def train(self):
        print('SVM: Training..')
        self.svclassifier = SVC(kernel='rbf', gamma='auto')
        self.model = self.svclassifier.fit(self.data.train_x_df, self.data.train_y_df)
        print('done..')

    def predict(self):
        print('SVM: Predicting..')
        y_pred = self.model.predict(self.data.test_x_df)
        rmse, mae, mape = Metrics.get_error(self.data.test_y_df, y_pred)
        sys.stdout.write(str(rmse))
        return y_pred, rmse, mae, mape

    def optimize(self):
        # defining parameter range
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf']}

        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
        # fitting the model for grid search
        grid.fit(self.data.train_x_df, self.data.train_y_df)

        # print best parameter after tuning
        print(grid.best_params_)
        # print how our model looks after hyper-parameter tuning
        print(grid.best_estimator_)

    def save(self, name):
        with open(name, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self, name):
        with open(name, 'rb') as file:
            self.model = pickle.load(file)


