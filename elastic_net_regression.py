from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

from tidy_tuesday import load_tidy_tuesday_dataset

# Uses Tidy Tuesday dataset for student debt stratified by race.
# The model predicts loan debt percentage based on previous years, student race, and loan debt.

df = load_tidy_tuesday_dataset(2021, 2, 9, 'student_debt')

x = df[['year', 'race', 'loan_debt']]
y = df['loan_debt_pct']
x = pd.get_dummies(x, columns=['race'], drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
elastic_net.fit(x_train, y_train)

y_pred_train = elastic_net.predict(x_train)
y_pred_test = elastic_net.predict(x_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f'Training MSE: {mse_train:.4f}')
print(f'Test MSE: {mse_test:.4f}')
print(f'Training R^2: {r2_train:.4f}')
print(f'Test R^2: {r2_test:.4f}')