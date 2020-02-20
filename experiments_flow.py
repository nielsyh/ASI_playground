from keras.models import load_model
import metrics
from data_frame_asi import DataFrameIMG
from tqdm import tqdm


def get_model(month, day):
    return load_model('cnn/' + str(month) + str(day) + '.h5')

def predict(data, model):
        y_pred =  model.predict(data.mega_df_x)
        rmse, mae, mape = metrics.Metrics.get_error(data.mega_df_y, y_pred)
        return y_pred, rmse, mae, mape

def experiment(prediction_horizon):
    prem = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    # prem = [(9,11)]
    for tup in tqdm(prem, total=len(prem), unit='Days to predict'):
        model = get_model(tup[0], tup[1])

        data = DataFrameIMG(False, prediction_horizon)
        data.build_img_df(6, 20, tup[0], tup[1])
        y_pred, rmse, mae, mape = predict(data, model)

        name = 'CNN_predhor' + str(prediction_horizon)
        metrics.Metrics.write_results(name, data.mega_df_times, data.mega_df_y, y_pred.ravel(), prediction_horizon)


# model = get_model(9,11)
# data = DataFrameIMG(False, 20)
# data.build_img_df(12,13,9,11)
# print(predict(data))
prediction_horizons = list(range(1,21))
for i in prediction_horizons:
    experiment(i)
    print('done: ' + str(i))

