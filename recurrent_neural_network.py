from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score

# Uses the IMDB dataset from Keras to predict sentiment towards a movie through reviews.
# We are predicting whether a review is positive or negative.

MAX_FEATURES = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top MAX_FEATURES most common words)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(MAX_FEATURES, 128, input_length=maxlen))
model.add(SimpleRNN(128, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

y_pred = (model.predict(x_test) > 0.5).astype("int32")

acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.4f}')
