import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Load and preprocess the dataset
data = pd.read_csv('Twitter_Data.csv')

# Convert 'clean_text' column to strings
data['clean_text'] = data['text'].astype(str)
data['clean_text'] = data['clean_text'].str.replace('[^a-zA-Z\s]', '', regex=True).str.lower()

# Split data into training and testing sets
X = data['clean_text']
y = data['sentiment']

# Ensure that your labels are numeric
y = y.replace({'negative': 0, 'neutral': 1, 'positive': 2})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and pad text sequences
tokenizer = Tokenizer(num_words=10000) # Increased the vocabulary size
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=200, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=200, padding='post')

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encode labels
num_classes = len(np.unique(y_train_encoded))
y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

# Build a more complex LSTM model with Bidirectional LSTM
model = tf.keras.Sequential([
Embedding(input_dim=10000, output_dim=100, input_length=200),
Bidirectional(LSTM(128, return_sequences=True)),
Dropout(0.5),
Bidirectional(LSTM(64)),
Dense(64, activation='relu'),
Dropout(0.5),
Dense(num_classes, activation='softmax')
])

# Compile the model with a learning rate scheduler
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
loss='categorical_crossentropy',
metrics=['accuracy'])

# Add a learning rate scheduler
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

# Train the model with an increased number of epochs and a learning rate scheduler
model.fit(X_train_pad, y_train_onehot, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test_onehot), callbacks=[lr_reduction])

# Save the trained model
model.save('sentiment_model.h5')
dump(tokenizer, 'tokenizer.joblib')


#add code to read sentiment_model.h5 model and predict values