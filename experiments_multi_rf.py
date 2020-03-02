from data.dataframe_sequence_multi import DataFrameSequenceMulti
from models.models_ts_multi import rf_model_multi
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from metrics import Metrics

start = 6
end = 19
prediction_horizons = list(range(1, 21))


def rf_experiment():
    permutations = [(True, True, True),(True, False, False), (False, True, False), (False, False, True)]
    permutations_names = ['all data', 'onsite_only', 'img only', 'meteor only']
    sqs = [30, 60, 120]
    for pidx, p in enumerate(permutations):
        for s in sqs:
            data = DataFrameSequenceMulti(False, p[0], p[1], p[2])
            data.build_ts_df(start,end,[7,8,9,10,11,12],s)
            data.normalize_mega_df()

            name_time = '_sqnc_' + str(sqs)
            name_data = 'data_' + permutations_names[pidx]

            rf = rf_model_multi.RF_predictor(data, 'RF SEQUENCE multi' + name_time + name_data)
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


# rf_experiment()
rf_test()

# rd_search_grid()
# minutes_sequence = int(sys.argv[1])
# cams = int(sys.argv[2])
# img = int(sys.argv[3])
#
# if img == 1:
#     img = True
# else:
#     img = False
#
# print('Minutes sequence: ' + str(minutes_sequence))
# print('cams: ' + str(cams))
# print('IMG: ' + str(img))
#
# rf_experiment(minutes_sequence, cams, img)
