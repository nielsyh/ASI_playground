import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

class Models:

    def __init__(self):
        pass

    def svm_model(self, x_train, y_train):
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(x_train, y_train)

    def print_res(self, y_test, y_pred):
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

