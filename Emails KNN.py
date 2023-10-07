from KNN import K_Nearest_Neighbors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def confusion_matrix(labels, predictions):
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

    for i in range(len(labels)):
        label = labels[i]
        prediction = predictions[i]
        if label == 1 and prediction == 1:
            true_positive += 1
        elif label == 0 and prediction == 0:
            true_negative += 1
        elif label == 0 and prediction == 1:
            false_positive += 1
        else:
            false_negative += 1
        
    return true_positive, true_negative, false_positive, false_negative

def five_fold_cross_validation(X, y, k = 1):
    accuracy, precision, recall = [0] * 5, [0] * 5, [0] * 5
    size = 1000
    
    for i in range(5):
        fold_index_start = i * size
        fold_index_end = (i+1) * size
        train_X = np.array(list(X)[:fold_index_start] + list(X)[fold_index_end:])
        train_y = np.array(list(y)[:fold_index_start] + list(y)[fold_index_end:])
        test_X = np.array(list(X)[fold_index_start:fold_index_end])
        test_y = np.array(list(y)[fold_index_start:fold_index_end])
        
        knn_model = K_Nearest_Neighbors(k, train_X, train_y)
        fold_predictions = knn_model.predictions(test_X)

        true_positive, true_negative, false_positive, false_negative = confusion_matrix(test_y, fold_predictions)
        accuracy[i] = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        precision[i] = (true_positive) / (true_positive + false_positive)
        recall[i] = (true_positive) / (true_positive + false_negative)
        print(f'K: {k}, Fold: {i}, Accuracy: {accuracy[i]}, Precision: {precision[i]}, Recall: {recall[i]}')
        
    return np.array(accuracy).mean(), np.array(precision).mean(), np.array(recall).mean()


def main():
    data = pd.read_csv('hw3Data/emails.csv', delimiter=',')
    X = np.array(data.iloc[:, 1:-1])
    y = np.array(data.iloc[:, -1])

    for k in [1, 3, 5, 7, 10] :
        accuracy, precision, recall = five_fold_cross_validation(X, y, k)
        print("Averages:", k, accuracy, precision, recall)

if __name__ == '__main__':
    main()