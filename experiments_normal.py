from data.dataframe_normal import DataFrameNormal
from data.dataframe_sequence import DataFrameSequence
from models import persistence_model, regression_model, svm_model, cnn_model, ann_model
import threading
import sys

arg = sys.argv
# 1 model 1=svm 2=ann 3=persistence
# 2 debug 1=yes, default no
# 3 prediction horizon 0 default


prediction_horizons = list(range(1, 21))
print(prediction_horizons)

def SVM_experiment_thread(prediction_horizon):
    print("Thread %s: starting", prediction_horizon)

    data = DataFrameSequence(meteor_data=True, images=True, debug=False)
    data.build_df(10, 17, 1, months=[7,8,9,10,11,12])
    data.set_prediction_horizon(prediction_horizon)
    svm = svm_model.SVM_predictor(data, model_name='SVM norm: default + images + metoer')
    svm.run_experiment()
    print("Thread %s: finishing", prediction_horizon)

def run_svm_multi_thread():
    for i in prediction_horizons:
        x = threading.Thread(target=SVM_experiment_thread, args=(i,))
        x.start()

data = DataFrameNormal(meteor_data=False, images=False, debug=False)
data.build_df(10, 14, 1, months=[8,9])
data.set_prediction_horizon(20)
svm = svm_model.SVM_predictor(data, model_name='test')

svm.data.split_data_set(9, 15)
svm.data.flatten_data_set()
svm.data.normalize_data_sets()
svm.train()
y_pred, rmse, mae, mape = svm.predict()
print(rmse)
print(y_pred)


