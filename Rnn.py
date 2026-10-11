import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocabulary_size = 10_000
max_length = 200
train_size = 20_000
test_size = 5_000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocabulary_size)

# Daha kısa diziler ve veri alt kümeleri eğitim süresini azaltır.
x_train = pad_sequences(
    x_train[:train_size], maxlen=max_length, padding="pre", truncating="pre"
)
y_train = y_train[:train_size]
x_test = pad_sequences(
    x_test[:test_size], maxlen=max_length, padding="pre", truncating="pre"
)
y_test = y_test[:test_size]

model = Sequential(
    [
        Embedding(input_dim=vocabulary_size, output_dim=32, mask_zero=True),
        SimpleRNN(32),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

model.fit(
    x_train,
    y_train,
    epochs=4,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=2,
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test kaybı: {loss:.4f}")
print(f"Test doğruluğu: {accuracy:.4f}")
