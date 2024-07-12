import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from tidy_tuesday import load_tidy_tuesday_dataset


# Uses Tidy Tuesday dataset for NHL player births.
# The model predicts games played based on birth year/month and country.

df = load_tidy_tuesday_dataset(2024, 1, 9, 'nhl_player_births')
# Add random data to simulate here since the data set didn't have a variable to predict
df['games_played'] = random.randint(100, 1000)

x = df[['birth_year', 'birth_country', 'birth_month']]
y = df['games_played']

x = pd.get_dummies(x, columns=['birth_country'], drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=100, random_state=42)
xg_reg.fit(x_train, y_train)

y_pred_train = xg_reg.predict(x_train)
y_pred_test = xg_reg.predict(x_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f'Training MSE: {mse_train:.4f}')
print(f'Test MSE: {mse_test:.4f}')
print(f'Training R^2: {r2_train:.4f}')
print(f'Test R^2: {r2_test:.4f}')