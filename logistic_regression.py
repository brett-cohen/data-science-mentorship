from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error, auc
import plotly.graph_objs as go

from tidy_tuesday import load_tidy_tuesday_dataset

# Uses Tidy Tuesday dataset on Bob Ross episode.
# Attempts to predict the number of color used based on season/episode number.

df = load_tidy_tuesday_dataset(2023, 2, 21, 'bob_ross')

X = df[['season', 'episode']]
y = df['num_colors']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)

mae_logistic = mean_absolute_error(y_test, y_pred_logistic)
mse_logistic = mean_squared_error(y_test, y_pred_logistic)

print(f'Logistic Regression MAE: {mae_logistic}')
print(f'Logistic Regression MSE: {mse_logistic}')

# SVM model
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

mae_svm = mean_absolute_error(y_test, y_pred_svm)
mse_svm = mean_squared_error(y_test, y_pred_svm)

print(f'SVM MAE: {mae_svm}')
print(f'SVM MSE: {mse_svm}')
