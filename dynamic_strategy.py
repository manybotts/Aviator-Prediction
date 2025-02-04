# Dynamic Strategy Switching

import logging
import os
import json

logger = logging.getLogger(__name__)

# Evaluate model performance
def evaluate_models():
    global model_performance

    # Calculate accuracy for each model
    accuracies = {}
    for model_name, results in model_performance.items():
        if results:
            correct = sum(1 for feedback, actual, predicted in results if feedback == "yes")
            total = len(results)
            accuracies[model_name] = correct / total if total > 0 else 0

    logger.info(f"Model Accuracies: {accuracies}")
    return accuracies


# Switch strategy based on performance
def switch_strategy(context):
    global current_strategy, model_performance

    accuracies = evaluate_models()

    # Find the best-performing model
    best_model = max(accuracies, key=accuracies.get)
    if accuracies[best_model] < 0.6:  # Switch if accuracy is below 60%
        logger.info("Accuracy below threshold. Switching strategy...")
        candidates = ["random_forest", "linear_regression", "lstm"]
        candidates.remove(current_strategy)  # Exclude the current strategy
        current_strategy = candidates[0]  # Switch to the first alternative

        # Notify users of the strategy change
        context.bot.send_message(
            chat_id=context.user_data["chat_id"],
            text=f"⚠️ The bot has switched to the '{current_strategy.capitalize()}' strategy to improve predictions."
        )

    logger.info(f"Current Strategy: {current_strategy}")


# Update model performance
def update_model_performance(feedback, actual_value, predicted_value, context):
    global model_performance

    # Add the result to the current model's performance history
    model_performance[current_strategy].append((feedback, actual_value, predicted_value))

    # Save performance data
    with open("model_performance.json", "w") as f:
        json.dump(model_performance, f)

    # Evaluate and switch strategy if necessary
    switch_strategy(context)
