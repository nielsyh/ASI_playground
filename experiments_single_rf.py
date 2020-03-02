from data.dataframe_sequence import DataFrameSequence
from models.models_ts import rf_model
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor


start = 6
end = 19
prediction_horizons = list(range(1, 21))


def rf_experiment():

    sqs = [30, 60, 120]
    stages = [1,2]
    for st in stages:
        for s in sqs:
            for i in prediction_horizons:
                print('start rf: ' + str(i))
                if st == 1:
                    data = DataFrameSequence(False,i, False, True)
                if st == 2:
                    data = DataFrameSequence(False, i, True, True)
                data.build_ts_df(start,end,[7,8,9,10,11,12],s, 1)
                data.normalize_mega_df()
                name_time = '_sqnc_' + str(i)
                name_cam = 'CAM_' + str(1)
                # name_img = '_img_' + str(img)
                name_stage = 'stg_' + str(st)
                name_pred = 'ph_' + str(i)
                rf = rf_model.RF_predictor(data, 'RF SEQUENCE' + name_time + name_cam + name_stage + name_pred)
                rf.set_days(data.get_thesis_test_days())
                rf.run_experiment()
                print('Finish rf: ' + str(i))

def rf_test():
    data = DataFrameSequence(True, 20, True, True)
    a = data.get_prem_days()
    data.build_ts_df(7, 10, [8,9], 5, 1, step=1)
    data.normalize_mega_df()
    rf = rf_model.RF_predictor(data, 'RF SEQUENCE PREM_ NO METOER')

    rf.set_days(a)

    data.split_data_set(9, 27)
    data.flatten_data_set()
    rf.train()
    y_pred, rmse, mae, mape = rf.predict()

    print(rmse)
    print(y_pred)

def rd_search_grid():
    data = DataFrameSequence(False, 20, True, True)
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
