# Aviator Prediction Telegram Bot with Reinforcement Learning

import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import re
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
from flask import Flask, request

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Store historical crash points and feedback
historical_data = []
feedback_data = []

# Load or train the ML model
def load_or_train_model():
    model_path = "random_forest_model.joblib"
    if os.path.exists(model_path):
        return load(model_path)
    else:
        # Train a new model if no saved model exists
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        dump(model, model_path)
        return model

model = load_or_train_model()

# Function to predict next outcome using ML
def predict_aviator_outcome(crash_points):
    global model

    # Convert crash points to numbers
    numbers = [float(point) for point in crash_points if re.match(r'^\d+(\.\d+)?$', point)]

    # Ensure there are enough data points
    count = len(numbers)
    if count == 0:
        return "No valid crash points found. Please provide numeric data."

    # Update the model with new data
    if count > 1:
        df = pd.DataFrame({"CrashPoints": numbers[:-1], "NextOutcome": numbers[1:]})
        X = df[["CrashPoints"]].values
        y = df["NextOutcome"].values
        model.fit(X, y)

        # Save the updated model
        dump(model, "random_forest_model.joblib")

    # Predict the next outcome
    last_point = np.array([[numbers[-1]]])
    predicted_outcome = model.predict(last_point)[0]

    # Output results
    result = (
        f"📊 Historical Crash Points: {', '.join(map(str, numbers))}\n"
        f"⚡ Predicted Next Outcome: {predicted_outcome:.2f}\n"
        f"❓ Was the prediction correct? Reply with 'yes' or 'no'."
    )
    return result, predicted_outcome


# Command handlers
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Welcome to the Aviator Prediction Bot! Enter crash points separated by commas to get predictions."
    )


def help_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        "To use this bot:\n"
        "- Enter crash points separated by commas (e.g., 2.5, 3.1, 4.0).\n"
        "- The bot will analyze the data and predict the next outcome.\n"
        "- Provide feedback ('yes' or 'no') to help the bot learn."
    )


def clear_data(update: Update, context: CallbackContext):
    global historical_data, feedback_data
    historical_data = []
    feedback_data = []
    update.message.reply_text("Historical data and feedback have been cleared.")


# Flask App for Gunicorn Compatibility
app = Flask(__name__)  # Define the Flask app here

@app.route('/<token>', methods=['POST'])
def webhook(token):
    global updater
    if updater and token == os.getenv("TELEGRAM_BOT_TOKEN"):
        # Pass the request body to the Telegram bot
        update = Update.de_json(request.json, updater.bot)
        updater.dispatcher.process_update(update)
    return '', 200


@app.route('/')
def index():
    return "Aviator Prediction Bot is running!", 200


# Telegram Bot Initialization
def init_bot():
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    PORT = int(os.getenv("PORT", "8443"))

    if not TOKEN:
        logger.error(" TELEGRAM_BOT_TOKEN environment variable is missing.")
        return None

    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Add handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("clear", clear_data))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, process_crash_points))
    dispatcher.add_handler(MessageHandler(Filters.regex(r"^(yes|no)$"), process_feedback))
    dispatcher.add_handler(MessageHandler(Filters.regex(r"^\d+(\.\d+)?$"), process_actual_value))
    dispatcher.add_error_handler(error_handler)

    return updater


updater = init_bot()  # Initialize the Telegram bot globally

# Main function
def main():
    if not updater:
        logger.error("Bot initialization failed. Exiting...")
        return

    HEROKU_APP_NAME = os.getenv("HEROKU_APP_NAME")
    if HEROKU_APP_NAME:
        HEROKU_URL = f"https://{HEROKU_APP_NAME}.herokuapp.com/"
        webhook_url = f"{HEROKU_URL}{os.getenv('TELEGRAM_BOT_TOKEN')}"
        logger.info(f"Starting webhook on {webhook_url}")
        updater.start_webhook(
            listen="0.0.0.0",
            port=int(os.getenv("PORT", "8443")),
            url_path=os.getenv("TELEGRAM_BOT_TOKEN"),
            webhook_url=webhook_url
        )
    else:
        logger.info("Starting polling mode (local testing)")
        updater.start_polling()

    updater.idle()


if __name__ == "__main__":
    main()
    
