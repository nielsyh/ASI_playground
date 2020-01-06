from abc import ABC, abstractmethod
from data import Data
from metrics import Metrics
import pickle


class Predictor_template(ABC):

    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def save(self, name):
        pass

    def load(self, name):
        pass

