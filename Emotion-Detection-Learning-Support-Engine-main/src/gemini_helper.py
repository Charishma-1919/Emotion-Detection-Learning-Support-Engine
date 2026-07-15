import os
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)


model = genai.GenerativeModel("gemini-2.0-flash")


def generate_learning_support(problem, emotion):

    prompt = f"""
You are an AI Learning Support Assistant.

Student Emotion: {emotion}

Student Problem: {problem}

Provide:
1. Empathetic response
2. Simple explanation
3. Learning tips
4. Next steps

Keep the response motivating and easy to understand.
"""

    try:
        response = model.generate_content(prompt)
        return response.text

    except Exception:
        return (
            f"I understand you are feeling {emotion}.\n\n"
            "Learning Tips:\n"
            "- Break the topic into smaller concepts.\n"
            "- Practice with simple examples.\n"
            "- Revise the basics and try again."
        )


def regenerate_response(problem, emotion):

    return generate_learning_support(problem, emotion)