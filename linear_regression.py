import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from tidy_tuesday import load_tidy_tuesday_dataset

# Using Tidy Tuesday bird count data as an example dataset
df = load_tidy_tuesday_dataset(2019, 6, 18, 'bird_counts')
df = df.dropna()

X = df[['how_many_counted']]
y = df['total_hours']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)