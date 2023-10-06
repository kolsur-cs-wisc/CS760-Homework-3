from LogisticRegression import Logistic_Regression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def five_fold_cross_validation(X, y):
    accuracy = [0 for _ in range(5)]
    size = 1000
    
    for i in range(5):
        fold_index_start = i * size
        fold_index_end = (i+1) * size
        train_X = np.array(list(X)[:fold_index_start] + list(X)[fold_index_end:])
        train_y = np.array(list(y)[:fold_index_start] + list(y)[fold_index_end:])
        test_X = np.array(list(X)[fold_index_start:fold_index_end])
        test_y = np.array(list(y)[fold_index_start:fold_index_end])
        
        logistic_model = Logistic_Regression(train_X, train_y)
        logistic_model.train()

        fold_preds = logistic_model.predict_labels(test_X)
        fold_error = abs(fold_preds - test_y)
        accuracy[i] = 1 - np.mean(fold_error)
        print(i, accuracy)
        
    return accuracy, np.mean(accuracy)

def main():
    data = pd.read_csv('hw3Data/emails.csv', delimiter=',')
    X = np.array(data.iloc[:, 1:-1])
    y = np.array(data.iloc[:, -1])

    print(five_fold_cross_validation(X, y))

if __name__ == '__main__':
    main()