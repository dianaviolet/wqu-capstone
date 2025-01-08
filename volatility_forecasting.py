import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from arch import arch_model

# Load the preprocessed data
train_data = pd.read_csv('Data/train_data.csv', index_col=0)
val_data = pd.read_csv('Data/val_data.csv', index_col=0)
test_data = pd.read_csv('Data/test_data.csv', index_col=0)


def create_sequences(data, seq_length, target_col_idx=0):
    """
    Create sequences for LSTM input with specific target column
    """
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        target = data[i + seq_length, target_col_idx]  # Only take the target column
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


# Model parameters
sequence_length = 20
batch_size = 32
epochs = 100
target_col_idx = 0  # Index of the target column (assuming first column is what we want to predict)

# Prepare sequences for LSTM
X_train, y_train = create_sequences(train_data.values, sequence_length, target_col_idx)
X_val, y_val = create_sequences(val_data.values, sequence_length, target_col_idx)
X_test, y_test = create_sequences(test_data.values, sequence_length, target_col_idx)


def build_lstm_model(input_shape, units=50):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Create and compile the model
model = build_lstm_model((sequence_length, X_train.shape[2]))

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)


def evaluate_model(model, X, y, set_name):
    """
    Evaluate the model and return predictions
    """
    predictions = model.predict(X).flatten()  # Flatten predictions to match target shape
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mse)

    print(f"\n{set_name} Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    return predictions


# Evaluate on test set
test_predictions = evaluate_model(model, X_test, y_test, "Test Set")

# Plot training history
plt.figure(figsize=(15, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot predictions vs actual
plt.subplot(1, 2, 2)
plt.plot(y_test[:100], label='Actual', alpha=0.8)
plt.plot(test_predictions[:100], label='Predicted', alpha=0.8)
plt.title('Volatility Forecast vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Volatility')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('Models/volatility_lstm_model.h5')

# Print shapes for verification
print("\nData Shapes:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"test_predictions shape: {test_predictions.shape}")