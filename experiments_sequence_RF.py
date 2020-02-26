from dataframe_sequence import DataFrameSequence
from metrics import Metrics
from models_ts import rf_model
import threading
import sys

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


start = 6
end = 19
prediction_horizons = list(range(1, 21))


def rf_experiment(minutes_sequence, cams,img):
    for i in prediction_horizons:
        print('start rf: ' + str(i))
        data = DataFrameSequence(False,i, False, img)
        data.build_ts_df(start,end,[7,8,9,10,11,12],minutes_sequence, cams)
        data.normalize_mega_df()
        name_time = '_sequence_' + str(minutes_sequence)
        name_cam = 'CAM_' + str(cams)
        name_img = '_img_' + str(img)
        name_pred = 'predhor_' + str(i)
        rf = rf_model.RF_predictor(data, 'RF SEQUENCE PREM_ NO METOER' + name_time + name_cam + name_img + name_pred)
        rf.run_experiment()
        print('Finish rf: ' + str(i))

def rd_search_grid():
    data = DataFrameSequence(False, 20, True, True)
    data.build_ts_df(6, 19, [7, 8, 9, 10, 11, 12], 60, 1)
    data.normalize_mega_df()

    rfr = RandomForestRegressor(bootstrap=True, random_state=0, n_jobs=-1, verbose=1)
    param_grid = dict(n_estimators=[50, 100, 200],
                      max_depth=[50,100,200],
                      min_samples_leaf=[1, 2, 4, 12, 24, 64])

    grid = GridSearchCV(rfr, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid.fit(data.test_x_df, data.test_y_df)
    print("grid.best_params_ {}".format(grid.best_params_))


rd_search_grid()
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
