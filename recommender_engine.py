import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve


# Used with Kaggle's online retail dataset: https://www.kaggle.com/datasets/nageshsingh/online-retail-dataset/data
df = pd.read_csv('online_retail_dataset.csv')

df['StockCode'] = df['StockCode'].astype(str)
customer_purchase_history = df.groupby('Customer ID').agg({
    'StockCode': lambda x: ' '.join(x),
    'InvoiceDate': 'max',
    'StockCode': 'last'
}).reset_index()

customer_purchase_history.columns = ['Customer ID', 'next_purchase', 'purchase_history']

x = customer_purchase_history['purchase_history']
y = customer_purchase_history['next_purchase']

vectorizer = CountVectorizer()
x_vectorized = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_vectorized, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Example predictions using stock codes picked at random
new_purchase_history = ['85048 22130 844064 85152 21819']
new_purchase_vectorized = vectorizer.transform(new_purchase_history)
predicted_item = model.predict(new_purchase_vectorized)

print(f"Predicted next purchase: {predicted_item[0]}")
