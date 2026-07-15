import numpy as np


# Five emotion classes according to project requirement
EMOTIONS = [
    "Bored",
    "Confident",
    "Confused",
    "Curious",
    "Frustrated"
]


def predict_emotion(text):
    """
    Emotion prediction function.
    Currently uses keyword enhancement.
    Later this can be connected with trained BiLSTM/BERT models.
    """

    text = text.lower()

    emotion_scores = {
        "Bored": 0,
        "Confident": 0,
        "Confused": 0,
        "Curious": 0,
        "Frustrated": 0
    }


    # Keyword based scoring
    keywords = {

        "Bored": [
            "boring",
            "tired",
            "not interested"
        ],

        "Confident": [
            "easy",
            "completed",
            "understand",
            "good"
        ],

        "Confused": [
            "confused",
            "don't understand",
            "stuck",
            "lost"
        ],

        "Curious": [
            "why",
            "how",
            "learn",
            "explore"
        ],

        "Frustrated": [
            "failed",
            "problem",
            "frustrated",
            "angry"
        ]
    }


    for emotion, words in keywords.items():
        for word in words:
            if word in text:
                emotion_scores[emotion] += 1


    # If no keyword found
    if max(emotion_scores.values()) == 0:
        emotion_scores["Curious"] = 1


    # Calculate confidence
    total = sum(emotion_scores.values())

    confidence = {}

    for emotion, score in emotion_scores.items():
        confidence[emotion] = round(
            (score / total) * 100,
            2
        )


    predicted = max(
        confidence,
        key=confidence.get
    )


    return {
        "emotion": predicted,
        "confidence": confidence
    }