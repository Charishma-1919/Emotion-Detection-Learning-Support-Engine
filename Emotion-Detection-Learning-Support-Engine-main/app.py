from src.dashboard import emotion_chart, confidence_chart
import streamlit as st

from src.emotion_detector import predict_emotion
from src.gemini_helper import generate_learning_support
from src.csv_logger import save_interaction


# Page configuration
st.set_page_config(
    page_title="Emotion Detection & Learning Support Engine",
    page_icon="😊",
    layout="wide"
)


# Session state
if "history" not in st.session_state:
    st.session_state.history = []


# Title
st.title("😊 Emotion Detection & Learning Support Engine")

st.write(
    "AI-powered platform that detects student emotions "
    "and provides personalized learning guidance."
)


# User input section
st.subheader("📚 Describe your learning problem")

problem = st.text_area(
    "Enter your study challenge:",
    placeholder="Example: I don't understand recursion and I feel stuck."
)


# Button
if st.button("Analyze Emotion"):

    if problem.strip() == "":
        st.warning("Please enter your learning problem.")

    else:

        # Emotion detection
        result = predict_emotion(problem)

        emotion = result["emotion"]
        confidence = result["confidence"]


        # Gemini response
        ai_response = generate_learning_support(
            problem,
            emotion
        )


        # Save history
        save_interaction(
            problem,
            emotion,
            confidence,
            ai_response
        )


        # Store session
        st.session_state.history.append(
            {
                "problem": problem,
                "emotion": emotion,
                "response": ai_response
            }
        )


        # Display result

        st.success("Emotion detected successfully!")


        st.subheader("😊 Detected Emotion")

        st.info(emotion)


        st.subheader("📊 Confidence Scores")

        st.json(confidence)


        st.subheader("📈 Emotion Confidence Visualization")

        chart = confidence_chart(confidence)

        if chart:
            st.plotly_chart(chart)


        st.subheader("🤖 AI Learning Support")

        st.write(ai_response)



# History section

st.divider()

st.subheader("📜 Previous Sessions")
st.subheader("📊 Overall Emotion Analytics")

history_chart = emotion_chart()

if history_chart:
    st.plotly_chart(history_chart)


for item in st.session_state.history:

    st.write(
        "Problem:",
        item["problem"]
    )

    st.write(
        "Emotion:",
        item["emotion"]
    )

    st.write(
        "Response:",
        item["response"]
    )

    st.divider()