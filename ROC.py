from KNN import K_Nearest_Neighbors
from LogisticRegression import Logistic_Regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd

def ROC_Emails():
    data = pd.read_csv('hw3Data/emails.csv', delimiter=',')
    X = np.array(data.iloc[:, 1:-1])
    y = np.array(data.iloc[:, -1])
    X_train, X_test = X[:4000, :], X[4000:, :]
    y_train, y_test = y[:4000], y[4000:]

    knn_model = K_Nearest_Neighbors(5, X_train, y_train)
    test_predictions_score = knn_model.predictions_prob(X_test)

    fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, test_predictions_score)
    auc_knn = roc_auc_score(y_test, test_predictions_score).round(2)
    print(f'AUC KNN: {auc_knn}')

    logistic_model = Logistic_Regression(X_train, y_train)
    logistic_model.train()
    test_predictions_score = logistic_model.predict_label_prob(X_test)

    fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(y_test, test_predictions_score)
    auc_logistic = roc_auc_score(y_test, test_predictions_score).round(2)
    print(f'AUC Logistic: {auc_logistic}')

    plt.plot(fpr_knn, tpr_knn, label = f'KNN, k = 5 (AUC = {auc_knn})')
    plt.plot(fpr_logistic, tpr_logistic, label = f'Logistic, (AUC = {auc_logistic})')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def Accuracy_KNN():
    points = np.array([[1, 0.8332],
    [3, 0.8423999999999999],
    [5, 0.8417999999999999],
    [7, 0.8454],
    [10, 0.8552]])
    
    plt.plot(points[:, 0], points[:, 1], marker = 'o')
    plt.xlabel("k")
    plt.ylabel("Average Accuracy")
    plt.title("kNN 5-Fold Cross Validation")
    plt.grid(True)
    plt.show()

def ROC_Given_Confidence():
    FPR = [0, 0, 1/4, 1/2, 1]
    TPR = [0, 1/3, 2/3, 1, 1]

    plt.plot(FPR, TPR, marker = 'o', clip_on = False)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.title("ROC Curve")
    plt.show()

def main():
    # ROC_Emails()
    Accuracy_KNN()
    # ROC_Given_Confidence()

if __name__ == '__main__':
    main()