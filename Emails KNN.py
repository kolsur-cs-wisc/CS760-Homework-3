from KNN import K_Nearest_Neighbors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def five_fold_cross_validation(X, y, k = 1):
    err = [0 for _ in range(5)]
    size = 1000
    
    for i in range(5):
        fold_index_start = i * size
        fold_index_end = (i+1)*size
        train_X = np.array(list(X)[:fold_index_start] + list(X)[fold_index_end:])
        train_y = np.array(list(y)[:fold_index_start] + list(y)[fold_index_end:])
        test_X = np.array(list(X)[fold_index_start:fold_index_end])
        test_y = np.array(list(y)[fold_index_start:fold_index_end])
        
        knn_model = K_Nearest_Neighbors(k, train_X, train_y)

        fold_preds = knn_model.predictions(test_X)
        fold_error = abs(fold_preds - test_y)
        err[i] = np.mean(fold_error)
        
    return err, np.mean(err)


def main():
    data = pd.read_csv('hw3Data/emails.csv', delimiter=',')
    X = np.array(data.iloc[:, 1:-1])
    y = np.array(data.iloc[:, -1])

    for k in [1, 3, 5, 7, 10] :
        result, avg = five_fold_cross_validation(X, y, k)
        print(k, result, avg)

if __name__ == '__main__':
    main()