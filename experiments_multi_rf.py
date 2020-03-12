from data.dataframe_sequence_multi import DataFrameSequenceMulti
from models.models_ts_multi import rf_model_multi
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from metrics import Metrics

start = 6
end = 19
sequence_lenth = 20

def run_final_all_days():
    data = DataFrameSequenceMulti(False, True, True, True)
    data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], sequence_lenth)
    data.normalize_mega_df()
    name_time = '_sqnc_' + str(sequence_lenth)
    name_data = 'data_' + 'all'
    rf = rf_model_multi.RF_predictor(data, 'RF SEQUENCE multi all data' + name_time + name_data)
    rf.set_days(data.get_all_test_days())
    rf.run_experiment()

def run_final_test_days():
    data = DataFrameSequenceMulti(False, True, True, True)
    data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], sequence_lenth)
    data.normalize_mega_df()
    name_time = '_sqnc_' + str(sequence_lenth)
    name_data = 'data_' + 'all'
    rf = rf_model_multi.RF_predictor(data, 'RF SEQUENCE multi testset' + name_time + name_data)
    rf.set_days(data.get_thesis_test_days())
    rf.run_experiment()

def rf_experiment():
    permutations = [(True, True, True)]
    permutations_names = ['all data']
    sqs = [5, 10, 20, 30]
    for pidx, p in enumerate(permutations):
        for s in sqs:
            data = DataFrameSequenceMulti(False, p[0], p[1], p[2])
            data.build_ts_df(start,end,[7,8,9,10,11,12],s)
            data.normalize_mega_df()

            name_time = '_sqnc_' + str(s)
            name_data = '_data_' + permutations_names[pidx]

            rf = rf_model_multi.RF_predictor(data, 'RF SEQUENCE multi prem' + name_time + name_data)
            rf.set_days(data.get_prem_days())
            rf.run_experiment()
            print('Finish rf')

def rf_test():
    data = DataFrameSequenceMulti(False, True, True, True)
    # a = data.get_prem_days()
    data.build_ts_df(7, 10, [8,9], 5)
    data.normalize_mega_df()
    rf = rf_model_multi.RF_predictor(data, 'RF SEQUENCE multi PREM_ NO METOER')
    # rf.set_days(a)
    data.split_data_set(9, 27)
    data.flatten_data_set()
    rf.train()
    y_pred, rmse, mae, mape = rf.predict()

    Metrics.write_results_multi('RF MULTI TEST', data.test_x_df.reshape(
        (data.test_x_df.shape[0],
         data.sequence_len_minutes,
         data.number_of_features)),
                              data.test_y_df, y_pred)

    print(rmse)
    print(y_pred)

def rd_search_grid():
    data = DataFrameSequenceMulti(False, True, True, True)
    data.build_ts_df(6, 19, [7, 8, 9, 10, 11, 12], 60, 1)
    data.normalize_mega_df()
    data.split_data_set(10, 15)
    data.flatten_data_set()

    rfr = RandomForestRegressor(bootstrap=True, random_state=0, n_jobs=-1, verbose=1)
    param_grid = dict(n_estimators=[50, 100, 200],
                      max_depth=[50,100,200],
                      min_samples_leaf=[1, 2, 4, 12, 24, 64])

    grid = GridSearchCV(rfr, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid.fit(data.test_x_df, data.test_y_df)
    print("grid.best_params_ {}".format(grid.best_params_))


# run_final_test_days()
rf_experiment()