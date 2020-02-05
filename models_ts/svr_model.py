from sklearn.svm import SVC
from dataframe_sequence import DataFrameSequence
from metrics import Metrics
import pickle
import calendar


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


        # self.day_month_to_predict = [(9,11)]

        for exp in self.day_month_to_predict:
            print('SVM: ' + str(exp) + ', horizon: ' + str(self.data.pred_horizon))
            self.data.split_data_set(exp[0], exp[1])
            self.data.flatten_data_set()
            self.train()
            y_pred, rmse, mae, mape = self.predict()

            Metrics.write_results(str(self.name), self.data.test_x_df, self.data.test_y_df, y_pred, self.data.pred_horizon)

    def train(self):
        print('SVM: Training..')
        self.svclassifier = SVC(kernel='rbf', gamma='auto')
        self.model = self.svclassifier.fit(self.data.train_x_df, self.data.train_y_df)
        print('done..')

    def predict(self):
        print('SVM: Predicting..')
        y_pred = self.model.predict(self.data.test_x_df)
        rmse, mae, mape = Metrics.get_error(self.data.test_y_df, y_pred)
        print(rmse)
        return y_pred, rmse, mae, mape

    def save(self, name):
        with open(name, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self, name):
        with open(name, 'rb') as file:
            self.model = pickle.load(file)


