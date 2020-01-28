from keras.layers import Input, Dense, concatenate, MaxPool2D, GlobalAveragePooling2D, Dropout, Conv2D, Flatten
from keras.models import Model
from sklearn.externals import joblib
import keras
from keras.applications.resnet50 import ResNet50
from keras.models import load_model

class resnet50:
    # windowSize minimum 32 for resnet-50
    def __init__(self, image_size, data):
        self.data = data
        self.model = 0

    def get_model(self, image_res):
        model = keras.models.Sequential()
        base = ResNet50(include_top=False, weights='imagenet',
                              input_shape=(image_res, image_res, 3))
        #freezeing all layers
        for layer in base.layers:
            layer.trainable = False
        print(base.summary())
        model.add(base)
        # model.add(Conv2D(64, kernel_size=3, input_shape=(224,224,3)))
        model.add(Flatten())
        model.add(Dense(100, kernel_initializer='normal'))
        model.add(Dense(10))
        model.add(Dense(1))
        print(model.summary())
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model


    def train(self, epochs=50, batch_size=128):
        self.model.fit(self.data.x_train, self.data.y_train, epochs=epochs, batch_size=batch_size)

    def predict(self):
        self.model.predict(self.data.x_test)

    def save_model(self, path):
        self.model.save(str(path) + '.h5')  # creates a HDF5 file 'my_model.h5'

    def load_model(self, path):
        self.model = load_model(str(path) + '.h5')