import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class KNN():
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

        predicted_labels = np.array([1 if np.count_nonzero(row[:self.k] == 1) >= self.k // 2 else 0 for row in dist_sorted_labels])
        return predicted_labels


def main():
    data = np.loadtxt('hw3Data/D2z.txt', delimiter=' ')
    X = np.array(data[:, 0:2])
    y = np.array(data[:, -1])

    range = np.round(np.arange(-2.0, 2.1, 0.1), 2)
    grid = np.array(np.meshgrid(range, range)).T.reshape(-1, 2)

    knn_model = KNN(1, X, y)
    predicted_labels = knn_model.predictions(grid)
    plt.scatter(grid[:, 0], grid[:, 1])
    plt.show()

if __name__ == '__main__':
    main()