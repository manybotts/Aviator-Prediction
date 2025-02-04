# Core Bot Logic

from telegram import Update  # Add this import
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)
import logging
import os

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Define conversation states
WAITING_FOR_CRASH_POINTS, WAITING_FOR_FEEDBACK, WAITING_FOR_ACTUAL_VALUE = range(3)

# Initialize global variables
historical_data = []
feedback_data = []

# Start command
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Welcome to the Aviator Prediction Bot! Enter crash points separated by commas to get predictions."
    )
    return WAITING_FOR_CRASH_POINTS


# Help command
def help_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        "To use this bot:\n"
        "- Enter crash points separated by commas (e.g., 2.5, 3.1, 4.0).\n"
        "- The bot will analyze the data and predict the next outcome.\n"
        "- Provide feedback ('yes' or 'no') to help the bot learn.\n"
        "- Use /stats to view the bot's learning history and improvement plan."
    )
    return WAITING_FOR_CRASH_POINTS


# Clear data command
def clear_data(update: Update, context: CallbackContext):
    global historical_data, feedback_data
    historical_data = []
    feedback_data = []
    context.user_data.clear()  # Clear user-specific data
    update.message.reply_text("Historical data and feedback have been cleared.")
    return WAITING_FOR_CRASH_POINTS


# Cancel command
def cancel(update: Update, context: CallbackContext):
    update.message.reply_text("Operation canceled. You can start over by entering crash points.")
    context.user_data.clear()  # Reset user state
    return WAITING_FOR_CRASH_POINTS


# Error handler
def error_handler(update: object, context: CallbackContext):
    logger.error(f"Update {update} caused error {context.error}")
    context.bot.send_message(chat_id=update.effective_chat.id, text="An unexpected error occurred. Please try again later.")


# Export conversation handler
def get_conversation_handler():
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WAITING_FOR_CRASH_POINTS: [
                MessageHandler(Filters.text & ~Filters.command, process_crash_points),
            ],
            WAITING_FOR_FEEDBACK: [
                MessageHandler(Filters.regex(r"^(yes|no)$"), process_feedback),
            ],
            WAITING_FOR_ACTUAL_VALUE: [
                MessageHandler(Filters.regex(r"^\d+(\.\d+)?$"), process_actual_value),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    return conv_handler
