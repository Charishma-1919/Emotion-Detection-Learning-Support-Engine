import pandas as pd
import os
import plotly.express as px


LOG_FILE = "logs/emotion_history.csv"


def load_history():

    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)

    return pd.DataFrame()



def emotion_chart():

    data = load_history()

    if data.empty:
        return None


    emotion_count = (
        data["Emotion"]
        .value_counts()
        .reset_index()
    )

    emotion_count.columns = [
        "Emotion",
        "Count"
    ]


    fig = px.bar(
        emotion_count,
        x="Emotion",
        y="Count",
        title="Emotion Distribution"
    )

    return fig



def confidence_chart(confidence):

    df = pd.DataFrame(
        confidence.items(),
        columns=[
            "Emotion",
            "Confidence"
        ]
    )


    fig = px.pie(
        df,
        names="Emotion",
        values="Confidence",
        title="Emotion Confidence Analysis"
    )

    return fig