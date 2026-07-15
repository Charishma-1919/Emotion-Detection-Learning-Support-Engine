import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (first time only)
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Cleans student input text before emotion detection
    """

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def keyword_emotion_detection(text):
    """
    Rule-based keyword enhancement
    """

    emotions = {
        "Confused": [
            "confused",
            "dont understand",
            "stuck",
            "lost",
            "difficult"
        ],
        "Frustrated": [
            "frustrated",
            "angry",
            "failed",
            "problem"
        ],
        "Curious": [
            "why",
            "how",
            "learn",
            "explore"
        ],
        "Confident": [
            "easy",
            "completed",
            "understand",
            "good"
        ],
        "Bored": [
            "boring",
            "tired",
            "not interested"
        ]
    }

    detected = []

    text = text.lower()

    for emotion, keywords in emotions.items():
        for word in keywords:
            if word in text:
                detected.append(emotion)
                break

    return detected