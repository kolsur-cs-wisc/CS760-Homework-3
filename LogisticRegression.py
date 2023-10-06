import math
import numpy as np

class Logistic_Regression():
    def __init__(self, X, y, alpha = 0.01, epoch = 10000):
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = y
        self.train_count, self.n = X.shape
        self.n += 1 #Add 1 for the bias term
        self.theta = np.zeros(self.n)
        self.alpha = alpha
        self.epoch = epoch

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))
    
    def gradient(self, predictions):
        return np.dot(self.X.T,(predictions - self.y)) / self.y.shape[0]
    
    def train(self):
        for _ in range(self.epoch):
            predictions = self.predict_sigmoid(self.X)
            grad = self.gradient(predictions)
            self.theta -= self.alpha * grad

    def predict_sigmoid(self, X_test):
        values = np.dot(self.theta, X_test.T)
        return self.sigmoid(values)
    
    def predict_labels(self, X_test):
        X_test_with_bias = np.column_stack((np.ones(X_test.shape[0]), X_test))
        values = self.predict_sigmoid(X_test_with_bias)
        return np.array([1 if curr_value >= 0.5 else 0 for curr_value in values])
    
    
    
    
