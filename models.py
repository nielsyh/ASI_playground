import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from data import Data

# class Models:
#
#     def __init__(self):
#         pass

def svm_model(x_train, y_train):
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(x_train, y_train)

def print_res(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

d = Data(pred_horzion=10, meteor_data=False)
d.build_df(7, 19, 1, months=[9])
d.label_df()
d.mega_df = d.mega_df.reshape(d.mega_df.shape[0]*d.mega_df.shape[1], -1)

# svclassifier = SVC(kernel='linear')
svclassifier = SVC(kernel='rbf')
x_train = d.mega_df[:,0:d.mega_df.shape[1]-1]
y_train = d.mega_df[:,-1]
svclassifier.fit(x_train, y_train)
