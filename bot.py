# Aviator Prediction Telegram Bot with Reinforcement Learning (Using Telegram Channel for Storage)

import logging
from telegram import Update, InputFile, Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import re
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
from flask import Flask, request
import requests

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Store historical crash points and feedback
historical_data = []
feedback_data = []

# Define the channel ID and file ID for the model
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
MODEL_FILE_ID = os.getenv("TELEGRAM_MODEL_FILE_ID")

# Load or train the ML model
def load_or_train_model(context):
    global MODEL_FILE_ID

    # Define the local model file path
    model_file = "random_forest_model.joblib"

    if MODEL_FILE_ID:
        try:
            # Initialize the Bot instance
            bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))

            # Get the file path from Telegram
            file_info = bot.get_file(MODEL_FILE_ID)
            logger.info(f"File Info: {file_info}")

            # Extract the file path
            file_path = file_info.file_path
            if not file_path:
                logger.error("File path is empty or invalid.")
                return regenerate_and_upload_model(context)

            # Construct the full file URL
            file_url = f"https://api.telegram.org/file/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/{file_path}"
            logger.info(f"Downloading model from: {file_url}")

            # Download the file using the requests library
            response = requests.get(file_url)
            if response.status_code == 200:
                with open(model_file, "wb") as f:
                    f.write(response.content)

                # Check if the file is valid
                if validate_model_file(model_file):
                    model = load(model_file)
                    logger.info("Model file loaded successfully.")
                    return model
                else:
                    logger.error("Downloaded model file is invalid or corrupted.")
                    return regenerate_and_upload_model(context)
            else:
                logger.error(f"Failed to download model file. Status code: {response.status_code}, Response: {response.text}")
                return regenerate_and_upload_model(context)
        except Exception as e:
            logger.error(f"Failed to download model from Telegram: {e}")
            return regenerate_and_upload_model(context)

    # If no model exists, train a new one
    logger.info("Training a new model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    dump(model, model_file)
    return model


# Function to validate the model file
def validate_model_file(file_path):
    if not os.path.exists(file_path):
        logger.error("Model file does not exist.")
        return False

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        logger.error("Model file is empty.")
        return False

    try:
        # Attempt to load the model
        model = load(file_path)
        logger.info("Model file validation successful.")
        return True
    except Exception as e:
        logger.error(f"Failed to validate model file: {e}")
        return False


# Function to regenerate and upload the model
def regenerate_and_upload_model(context):
    global MODEL_FILE_ID

    logger.info("Regenerating and uploading a new model...")

    # Train a new model
    model_file = "random_forest_model.joblib"
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    dump(model, model_file)

    # Upload the new model to Telegram
    try:
        bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
        with open(model_file, "rb") as f:
            sent_file = bot.send_document(chat_id=CHANNEL_ID, document=f)
            MODEL_FILE_ID = sent_file.document.file_id
            logger.info(f"New model uploaded successfully. File ID: {MODEL_FILE_ID}")
        return model
    except Exception as e:
        logger.error(f"Failed to upload new model to Telegram: {e}")
        logger.warning("Using default model for now.")
        return model


# Function to predict next outcome using ML
def predict_aviator_outcome(crash_points):
    global model

    # Convert crash points to numbers
    numbers = [float(point) for point in crash_points if re.match(r'^\d+(\.\d+)?$', point)]

    # Ensure there are enough data points
    count = len(numbers)
    if count == 0:
        return "No valid crash points found. Please provide numeric data.", None

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


# Function to update the model based on feedback
def update_model_with_feedback(feedback, actual_value, predicted_value, context):
    global model, MODEL_FILE_ID

    if feedback.lower() == "yes":
        logger.info("Feedback: Prediction was correct.")
    elif feedback.lower() == "no":
        logger.info("Feedback: Prediction was incorrect. Updating model...")

        # Add the actual value to the dataset
        historical_data.append(str(actual_value))

        # Retrain the model with the updated dataset
        numbers = [float(point) for point in historical_data]
        if len(numbers) > 1:
            df = pd.DataFrame({"CrashPoints": numbers[:-1], "NextOutcome": numbers[1:]})
            X = df[["CrashPoints"]].values
            y = df["NextOutcome"].values
            model.fit(X, y)

            # Save the updated model
            dump(model, "random_forest_model.joblib")

            # Upload the updated model to the Telegram channel
            try:
                bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
                with open("random_forest_model.joblib", "rb") as f:
                    sent_file = bot.send_document(chat_id=CHANNEL_ID, document=f)
                    MODEL_FILE_ID = sent_file.document.file_id
                    logger.info("Updated model file uploaded to Telegram channel.")
            except Exception as e:
                logger.error(f"Failed to upload model to Telegram: {e}")
    else:
        logger.warning("Invalid feedback received.")


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
    context.user_data.clear()  # Clear user-specific data
    update.message.reply_text("Historical data and feedback have been cleared.")


# Message handler for crash points
def process_crash_points(update: Update, context: CallbackContext):
    global historical_data

    # Reset user state
    context.user_data["state"] = "waiting_for_feedback"

    # Extract crash points from message
    input_data = update.message.text.strip()
    new_crash_points = [point.strip() for point in input_data.split(",") if point.strip()]

    # Add new crash points to historical data
    historical_data.extend(new_crash_points)

    # Predict next outcome
    prediction_message, predicted_value = predict_aviator_outcome(historical_data)
    update.message.reply_text(prediction_message)

    # Store the predicted value for later feedback
    context.user_data["predicted_value"] = predicted_value


# Message handler for feedback
def process_feedback(update: Update, context: CallbackContext):
    feedback = update.message.text.strip().lower()
    predicted_value = context.user_data.get("predicted_value")

    if predicted_value is None:
        update.message.reply_text("No recent prediction found. Please enter crash points first.")
        return

    if feedback not in ["yes", "no"]:
        update.message.reply_text("Invalid feedback. Please reply with 'yes' or 'no'.")
        return

    # Transition to waiting for actual value if feedback is 'no'
    if feedback == "yes":
        update.message.reply_text("Thank you for your feedback! The model will continue learning.")
        context.user_data.clear()  # Reset user state
    elif feedback == "no":
        update.message.reply_text("Please provide the actual value for the last prediction:")
        context.user_data["state"] = "waiting_for_actual_value"


# Handle actual value input
def process_actual_value(update: Update, context: CallbackContext):
    actual_value = update.message.text.strip()
    feedback = context.user_data.get("state")
    predicted_value = context.user_data.get("predicted_value")

    if feedback != "waiting_for_actual_value":
        update.message.reply_text("Unexpected input. Please follow the bot's instructions.")
        return

    if not re.match(r'^\d+(\.\d+)?$', actual_value) or float(actual_value) <= 0:
        update.message.reply_text("Invalid actual value. Please provide a positive number.")
        return

    # Update the model with feedback
    update_model_with_feedback("no", float(actual_value), predicted_value, context)

    update.message.reply_text("Thank you for your feedback! The model has been updated.")
    context.user_data.clear()  # Reset user state


# Error handler
def error_handler(update: object, context: CallbackContext):
    logger.error(f"Update {update} caused error {context.error}")
    context.bot.send_message(chat_id=update.effective_chat.id, text="An unexpected error occurred. Please try again later.")


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
    CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
    MODEL_FILE_ID = os.getenv("TELEGRAM_MODEL_FILE_ID")

    if not TOKEN or not CHANNEL_ID or not MODEL_FILE_ID:
        logger.error("Missing required environment variables.")
        return None

    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Load or train the model
    global model
    model = load_or_train_model(dispatcher.bot)

    # Add handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("clear", clear_data))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, process_crash_points))
    dispatcher.add_handler(MessageHandler(Filters.regex(r"^(yes|no)$"), process_feedback))
    dispatcher.add_handler(MessageHandler(Filters.regex(r"^\d+(\.\d+)?$"), process_actual_value))

    # Add error handler
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
