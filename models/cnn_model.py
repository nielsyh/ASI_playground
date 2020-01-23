from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.layers import Input, Dense, concatenate, MaxPool2D, GlobalAveragePooling2D
from keras.models import Model
from sklearn.externals import joblib
import keras
from keras.applications.resnet50 import ResNet50

class resnet50:
    # windowSize minimum 32 for resnet-50
    def __init__(self, image_size, path=None):
        pass


    def get_model(self, image_res):

        model = keras.models.Sequential()

        base = ResNet50(include_top=False, weights='imagenet',
                              input_shape=(image_res, image_res, 3))

        #freezeing all layers
        for layer in base.Layers:
            layer.trainable = False

        print(base.summary())

        model.add(base)
        # model.add(Conv2D(64, kernel_size=3, input_shape=(224,224,3)))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(10))
        model.add(Dense(1, kernel_initializer='normal'))
        print(model.summary())
        model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, model, data, labels, epochs, batch_size):
        model.fit(data, labels, epochs=epochs, batch_size=batch_size)

    def predict(self, model, x_test, y_test):
        return model.predict(x_test)


    def save_model(self, path):
        pass

    def load_model(self, path):
        pass