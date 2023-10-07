from KNN import K_Nearest_Neighbors
from LogisticRegression import Logistic_Regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd

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