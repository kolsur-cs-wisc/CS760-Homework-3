from KNN import K_Nearest_Neighbors
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(train_X, train_y, test_grid, predicted_labels):

    true_zero = train_X[train_y == 0]
    true_one = train_X[train_y == 1]
    pred_zero = test_grid[predicted_labels == 0]
    pred_one = test_grid[predicted_labels == 1]

    plt.scatter(pred_zero[:, 0], pred_zero[:, 1], s = 24, label='Predicted 0')
    plt.scatter(pred_one[:, 0], pred_one[:, 1], s = 24, label='Predicted 1')
    plt.scatter(true_zero[:, 0], true_zero[:, 1], marker = 'o', facecolors = 'none', edgecolors = 'black', label = 'True 0')
    plt.scatter(true_one[:, 0], true_one[:, 1], c = 'black', marker = '+', label = 'True 1')
    plt.legend()
    plt.show()


def main():
    data = np.loadtxt('hw3Data/D2z.txt', delimiter=' ')
    X = np.array(data[:, 0:2])
    y = np.array(data[:, -1])

    range = np.round(np.arange(-2.0, 2.1, 0.1), 2)
    grid = np.array(np.meshgrid(range, range)).T.reshape(-1, 2)

    knn_model = K_Nearest_Neighbors(1, X, y)
    predicted_labels = knn_model.predictions(grid)
    scatter_plot(X, y, grid, predicted_labels)

if __name__ == '__main__':
    main()