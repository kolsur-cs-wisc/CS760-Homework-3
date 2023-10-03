import numpy as np
import pandas as pd

class KNN():
    def __init__(self, k, X, y):
        self.k = k
        self.X = X
        self.y = y

    def euclidean(point1, point2):
        return np.sqrt(np.sum(np.square(np.subtract(point2, point1))), axis=1)
