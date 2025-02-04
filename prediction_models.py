# Prediction Models

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from joblib import dump, load
import os

# Initialize models
models = {
    "random_forest": None,
    "linear_regression": None,
    "lstm": None
}

current_strategy = "random_forest"
model_performance = {"random_forest": [], "linear_regression": [], "lstm": []}


# Load or train models
def load_or_train_models():
    global models

    # Train Random Forest
    if not os.path.exists("random_forest_model.joblib"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        dump(model, "random_forest_model.joblib")
    models["random_forest"] = load("random_forest_model.joblib")

    # Train Linear Regression
    if not os.path.exists("linear_regression_model.joblib"):
        model = LinearRegression()
        dump(model, "linear_regression_model.joblib")
    models["linear_regression"] = load("linear_regression_model.joblib")

    # Train LSTM
    if not os.path.exists("lstm_model.h5"):
        lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, 1)),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.save("lstm_model.h5")
    models["lstm"] = load_model("lstm_model.h5")


# Function to predict next outcome using the current strategy
def predict_aviator_outcome(crash_points):
    global models, current_strategy

    # Convert crash points to numbers
    numbers = [float(point) for point in crash_points if re.match(r'^\d+(\.\d+)?$', point)]

    # Ensure there are enough data points
    count = len(numbers)
    if count == 0:
        return "No valid crash points found. Please provide numeric data.", None

    # Update the current model with new data
    if count > 1:
        df = pd.DataFrame({"CrashPoints": numbers[:-1], "NextOutcome": numbers[1:]})
        X = df[["CrashPoints"]].values.reshape(-1, 1)
        y = df["NextOutcome"].values

        # Retrain the current model
        if current_strategy == "random_forest":
            models["random_forest"].fit(X, y)
            dump(models["random_forest"], "random_forest_model.joblib")
        elif current_strategy == "linear_regression":
            models["linear_regression"].fit(X, y)
            dump(models["linear_regression"], "linear_regression_model.joblib")
        elif current_strategy == "lstm":
            X_lstm = np.array([[x] for x in X]).reshape(-1, 1, 1)
            y_lstm = np.array(y).reshape(-1, 1)
            models["lstm"].fit(X_lstm, y_lstm, epochs=10, batch_size=1, verbose=0)
            models["lstm"].save("lstm_model.h5")

    # Predict the next outcome
    last_point = np.array([[numbers[-1]]])
    if current_strategy == "random_forest":
        predicted_outcome = models["random_forest"].predict(last_point)[0]
    elif current_strategy == "linear_regression":
        predicted_outcome = models["linear_regression"].predict(last_point)[0]
    elif current_strategy == "lstm":
        last_point_lstm = last_point.reshape(1, 1, 1)
        predicted_outcome = models["lstm"].predict(last_point_lstm)[0][0]

    # Output results
    result = (
        f"📊 Historical Crash Points: {', '.join(map(str, numbers))}\n"
        f"⚡ Predicted Next Outcome: {predicted_outcome:.2f}\n"
        f"🎯 Current Strategy: {current_strategy.capitalize()}\n"
        f"❓ Was the prediction correct? Reply with 'yes' or 'no'."
    )
    return result, predicted_outcome
