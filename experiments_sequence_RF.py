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

minutes_sequence = int(sys.argv[1])
cams = int(sys.argv[2])
img = int(sys.argv[3])

if img == 1:
    img = True
else:
    img = False

print('Minutes sequence: ' + str(minutes_sequence))
print('cams: ' + str(cams))
print('IMG: ' + str(img))

rf_experiment(minutes_sequence, cams, img)
