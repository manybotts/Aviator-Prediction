# Aviator Prediction Telegram Bot

import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import re
import os
from flask import Flask, request

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Store historical crash points
historical_data = []

# Function to analyze crash points and predict next outcome
def predict_aviator_outcome(crash_points):
    # Convert crash points to numbers
    numbers = [float(point) for point in crash_points if re.match(r'^\d+(\.\d+)?$', point)]

    # Ensure there are enough data points
    count = len(numbers)
    if count == 0:
        return "No valid crash points found. Please provide numeric data."

    # Calculate basic statistics
    average = sum(numbers) / count
    std_dev = (sum((x - average) ** 2 for x in numbers) / count) ** 0.5

    # Weighted Average (More weight to recent data)
    weighted_average = 0
    total_weight = 0
    for i in range(count):
        weight = count - i  # More weight to recent data
        weighted_average += numbers[i] * weight
        total_weight += weight
    weighted_average /= total_weight

    # Exponential Smoothing (Adaptive prediction)
    alpha = 0.5  # Smoothing factor (adjust as needed)
    exponential_smooth = numbers[-1]
    for i in range(count - 2, -1, -1):
        exponential_smooth = (alpha * numbers[i]) + ((1 - alpha) * exponential_smooth)

    # Combine predictions
    predicted_outcome = (weighted_average + exponential_smooth) / 2 * 0.8  # Slightly below average for safety

    # Output results
    result = (
        f"📊 Historical Crash Points: {', '.join(map(str, numbers))}\n"
        f"🔍 Average Crash Point: {average:.2f}\n"
        f"⚖️ Standard Deviation: {std_dev:.2f}\n"
        f"🎯 Weighted Average: {weighted_average:.2f}\n"
        f"🎉 Exponential Smoothed Prediction: {exponential_smooth:.2f}\n"
        f"⚡ Predicted Next Outcome: {predicted_outcome:.2f}"
    )
    return result


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
        "- Type /clear to reset the historical data."
    )


def clear_data(update: Update, context: CallbackContext):
    global historical_data
    historical_data = []
    update.message.reply_text("Historical data has been cleared.")


# Message handler
def process_crash_points(update: Update, context: CallbackContext):
    global historical_data

    # Extract crash points from message
    input_data = update.message.text.strip()
    new_crash_points = [point.strip() for point in input_data.split(",") if point.strip()]

    # Add new crash points to historical data
    historical_data.extend(new_crash_points)

    # Predict next outcome
    prediction = predict_aviator_outcome(historical_data)
    update.message.reply_text(prediction)


# Error handler
def error_handler(update: object, context: CallbackContext):
    logger.error(f"Update {update} caused error {context.error}")
    context.bot.send_message(chat_id=update.effective_chat.id, text="An unexpected error occurred.")


# Initialize the bot
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
    dispatcher.add_error_handler(error_handler)

    return updater


# Flask App for Gunicorn Compatibility
app = Flask(__name__)
updater = init_bot()

@app.route('/<token>', methods=['POST'])
def webhook(token):
    if updater and token == os.getenv("TELEGRAM_BOT_TOKEN"):
        # Pass the request body to the Telegram bot
        update = Update.de_json(request.json, updater.bot)
        updater.dispatcher.process_update(update)
    return '', 200


@app.route('/')
def index():
    return "Aviator Prediction Bot is running!", 200


if __name__ == "__main__":
    if updater:
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
