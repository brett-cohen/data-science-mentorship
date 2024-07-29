from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error, auc
import plotly.graph_objs as go
from tidy_tuesday import load_tidy_tuesday_dataset

# Uses Tidy Tuesday dataset on Bob Ross episode.
# Attempts to predict if season/episode number has an impact on number of colors used.
# e.g. Do later seasons use more colors?

df = load_tidy_tuesday_dataset(2023, 2, 21, 'bob_ross')
df['many_colors'] = (df['num_colors'] > 9).astype(int)

X = df[['season', 'episode']]
y = df['many_colors']

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

# AUROC
y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_prob_logistic)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)

roc_auc_logistic = auc(fpr_logistic, tpr_logistic)
roc_auc_svm = auc(fpr_svm, tpr_svm)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=fpr_logistic, y=tpr_logistic,
    mode='lines',
    name=f'Logistic Regression (AUROC = {roc_auc_logistic:.2f})'
))

fig.add_trace(go.Scatter(
    x=fpr_svm, y=tpr_svm,
    mode='lines',
    name=f'SVM (AUROC = {roc_auc_svm:.2f})'
))

fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    line=dict(dash='dash'),
    name='Random Guess'
))

fig.update_layout(
    title='Receiver Operating Characteristic (ROC) Curve',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    showlegend=True
)

fig.show()
