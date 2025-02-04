# Main Script

from flask import Flask, request
from core_bot import get_conversation_handler, start, help_command, clear_data, error_handler
from prediction_models import load_or_train_models, predict_aviator_outcome
from dynamic_strategy import update_model_performance
from learning_stats import stats_command
import os
import logging

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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


# Process crash points
def process_crash_points(update: Update, context: CallbackContext):
    global historical_data

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
    context.user_data["chat_id"] = update.effective_chat.id
    return WAITING_FOR_FEEDBACK


# Process feedback
def process_feedback(update: Update, context: CallbackContext):
    feedback = update.message.text.strip().lower()
    predicted_value = context.user_data.get("predicted_value")

    if predicted_value is None:
        update.message.reply_text("No recent prediction found. Please enter crash points first.")
        return WAITING_FOR_CRASH_POINTS

    if feedback not in ["yes", "no"]:
        update.message.reply_text("Invalid feedback. Please reply with 'yes' or 'no'.")
        return WAITING_FOR_FEEDBACK

    if feedback == "yes":
        update.message.reply_text("Thank you for your feedback! The model will continue learning.")
        context.user_data.clear()  # Reset user state
        return WAITING_FOR_CRASH_POINTS

    elif feedback == "no":
        update.message.reply_text("Please provide the actual value for the last prediction:")
        return WAITING_FOR_ACTUAL_VALUE


# Process actual value
def process_actual_value(update: Update, context: CallbackContext):
    actual_value = update.message.text.strip()
    predicted_value = context.user_data.get("predicted_value")

    if not re.match(r'^\d+(\.\d+)?$', actual_value) or float(actual_value) <= 0:
        update.message.reply_text("Invalid actual value. Please provide a positive number.")
        return WAITING_FOR_ACTUAL_VALUE

    # Update model performance
    update_model_performance("no", float(actual_value), predicted_value, context)

    update.message.reply_text("Thank you for your feedback! The model has been updated.")
    context.user_data.clear()  # Reset user state
    return WAITING_FOR_CRASH_POINTS


# Telegram Bot Initialization
def init_bot():
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    PORT = int(os.getenv("PORT", "8443"))
    CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

    if not TOKEN or not CHANNEL_ID:
        logger.error("Missing required environment variables.")
        return None

    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Load or train models
    load_or_train_models(dispatcher)

    # Add handlers
    conv_handler = get_conversation_handler()
    dispatcher.add_handler(conv_handler)
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("clear", clear_data))
    dispatcher.add_handler(CommandHandler("stats", stats_command))
    dispatcher.add_error_handler(error_handler)

    return updater


# Main function
if __name__ == "__main__":
    from core_bot import start
    from prediction_models import predict_aviator_outcome
    from dynamic_strategy import update_model_performance
    from learning_stats import stats_command

    updater = init_bot()

    if not updater:
        logger.error("Bot initialization failed. Exiting...")
    else:
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
