import csv
import os
from datetime import datetime


LOG_FILE = "logs/emotion_history.csv"


def save_interaction(
    user_input,
    emotion,
    confidence,
    response
):
    """
    Saves user interaction history into CSV file.
    """

    # Create logs folder if not available
    os.makedirs("logs", exist_ok=True)


    file_exists = os.path.isfile(LOG_FILE)


    with open(
        LOG_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)


        # Create header first time
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "User Input",
                "Emotion",
                "Confidence",
                "AI Response"
            ])


        writer.writerow([
            datetime.now(),
            user_input,
            emotion,
            confidence,
            response
        ])