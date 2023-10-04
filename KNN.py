import matplotlib.pyplot as plt
import numpy as np

class K_Nearest_Neighbors():
    def __init__(self, k, X, y):
        self.k = k
        self.X = X
        self.y = y

    def euclidean(self, point):
        return np.sqrt(np.sum(np.square(np.subtract(point, self.X)), axis=1))
    
    def predictions(self, X_test):
        distances = [self.euclidean(test_features) for test_features in X_test]
        dist_sorted_labels = []
        for i in range(len(distances)):
            dist_sorted_labels.append([y for _, y in sorted(zip(distances[i], self.y))])

        predicted_labels = np.array([1 if np.count_nonzero(sorted_labels_curr[:self.k]) > self.k // 2 else 0 for sorted_labels_curr in dist_sorted_labels])
        return predicted_labels