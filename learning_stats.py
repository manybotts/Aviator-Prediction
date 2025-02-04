# Learning History and Stats

from telegram import Update  # Add this import
from telegram.ext import CallbackContext  # Add this import
import json

# Get learning stats
def get_learning_stats():
    global model_performance

    stats = "📊 Bot Learning History:\n\n"

    # Calculate accuracy for each model
    accuracies = {}
    for model_name, results in model_performance.items():
        if results:
            correct = sum(1 for feedback, actual, predicted in results if feedback == "yes")
            total = len(results)
            accuracies[model_name] = correct / total if total > 0 else 0

    # Add accuracy stats to the message
    for model_name, accuracy in accuracies.items():
        stats += f"✅ {model_name.capitalize()} Accuracy: {accuracy * 100:.2f}%\n"

    # Add improvement plan
    stats += "\n🤖 Improvement Plan:\n"
    stats += "- Continuously evaluate model performance.\n"
    stats += "- Switch strategies if accuracy drops below 60%.\n"
    stats += "- Retrain models with new data to adapt."

    return stats


# Handle /stats command
def stats_command(update: Update, context: CallbackContext):
    stats = get_learning_stats()
    update.message.reply_text(stats)
