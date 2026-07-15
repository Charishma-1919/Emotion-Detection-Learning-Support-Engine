import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from emotion_pipeline import EmotionDetectionPipeline

# Load environmental variables
load_dotenv()

class GuidanceEngine:
    def __init__(self):
        # 1. Initialize the Core Emotion Pipeline (Epic 3)
        self.pipeline = EmotionDetectionPipeline()
        
        # 2. Initialize Gemini Client (Epic 1 environment configurations)
        try:
            # Automatically reads GEMINI_API_KEY from environment/.env
            self.client = genai.Client()
            self.api_available = True
            print("Gemini API Client initialized successfully.")
        except Exception as e:
            print(f"Gemini API loading error: {e}. Defaulting to Static Template Mode.")
            self.api_available = False

        # 3. Subtask 4: Session History & Logging configurations
        self.session_history = []
        self.log_path = os.path.join("data", "guidance_logs.csv")
        os.makedirs("data", exist_ok=True)

        # Subtask 2: Fallback Static Templates for different categories
        self.fallback_templates = {
            "joy": {
                "Academic": "That is wonderful news! Keep up this incredible academic momentum and celebrate your progress.",
                "Career": "Congratulations on this professional milestone! Use this positive momentum to set your next career goals.",
                "Personal": "It is great to feel joyful. Spend some time reflecting on what brought you this happiness.",
                "default": "Wonderful! Keep cultivating and sharing this positive energy."
            },
            "sadness": {
                "Academic": "Academic setbacks can feel heavy. Remember that learning is a journey with ups and downs. Take a break, and break your goals into small steps.",
                "Career": "Professional setbacks are difficult but temporary. Be kind to yourself, focus on what you can control, and don't hesitate to seek mentorship.",
                "Personal": "It is completely natural to experience sadness. Allow yourself space to process these feelings, and focus on gentle self-care today.",
                "default": "I hear you, and it's okay to feel sad. Please take extra care of yourself right now."
            },
            "anger": {
                "Academic": "Academic pressure can easily trigger frustration. Step away from your workspace for a short walk to clear your mind before trying again.",
                "Career": "Workplace friction can be highly irritating. Try addressing the issue objectively once the initial surge of frustration settles down.",
                "Personal": "Anger is a valid reaction, but carrying it can be exhausting. Consider journaling or physical activity as a constructive outlet.",
                "default": "Take a slow, deep breath. Let's work on channeling this energy into constructive steps."
            },
            "fear": {
                "Academic": "Exam or assignment anxiety is very common. Focus your mind purely on the next immediate task, and trust the work you have already put in.",
                "Career": "Career changes or heavy professional duties can feel daunting. Treat large tasks as smaller individual projects to build confidence.",
                "Personal": "When facing worry, bring your focus back to the present moment. Reach out to a loved one if you feel overwhelmed.",
                "default": "It's normal to feel anxious or uncertain. Take things one slow, deliberate step at a time."
            },
            "love": {
                "Academic": "Your genuine passion for learning is wonderful. Keep fostering this positive energy and curiosity in your studies.",
                "Career": "Finding meaning and passion in your work is a rare gift. Continue aligning your professional path with your core values.",
                "Personal": "Experiencing love and appreciation is beautiful. Take a moment to express your gratitude to the connections that support you.",
                "default": "That is beautiful. Nourish and cherish these meaningful connections."
            }
        }

    # =======================================================
    # Subtask 1: Build Gemini Prompt with Emotion/Confidence
    # =======================================================
    def build_prompt(self, field: str, problem_description: str, emotion_result: dict) -> str:
        primary = emotion_result['primary_emotion']
        confidence = emotion_result['primary_confidence'] * 100
        
        # Compile mixed-emotion context if present
        mixed_context = ""
        if emotion_result['is_mixed']:
            sec_em = emotion_result['secondary_emotion']
            sec_conf = emotion_result['secondary_confidence'] * 100
            mixed_context = f"A secondary emotion of '{sec_em}' was also detected with {sec_conf:.1f}% probability."

        prompt = f"""
        You are an empathetic, constructive AI Life Guide. 
        A user has presented a problem within the '{field}' domain.
        
        User Problem Statement: "{problem_description}"
        
        Emotion Analytics detected:
        - Primary Emotion: {primary} ({confidence:.1f}% probability)
        {mixed_context}
        
        Generate an actionable, comforting response satisfying these guidelines:
        1. Empathize directly with their emotional state based on the detected emotion and confidence.
        2. Deliver field-aware, practical guidance matching the '{field}' category.
        3. Do not output internal metadata, technical ML terms (like logits, Softmax, BERT, or BiLSTM), or explain your code rules to the user.
        
        Keep your output warm, encouraging, structured, and under 150 words.
        """
        return prompt

    # =======================================================
    # Subtask 2: Generate Responses (With Fallback Mechanism)
    # =======================================================
    def generate_response(self, field: str, problem_description: str, emotion_result: dict) -> str:
        # Failsafe check
        if not self.api_available:
            return self.fallback_to_template(field, emotion_result)
            
        prompt = self.build_prompt(field, problem_description, emotion_result)
        
        try:
            # Query Gemini API (Using modern google-genai standards)
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini generation failed: {e}. Reverting to fallback static template.")
            return self.fallback_to_template(field, emotion_result)

    def fallback_to_template(self, field: str, emotion_result: dict) -> str:
        primary_emotion = emotion_result['primary_emotion']
        # Check if the user's field is covered in our template keys
        field_key = field if field in ["Academic", "Career", "Personal"] else "default"
        
        emotion_set = self.fallback_templates.get(primary_emotion, {})
        fallback_msg = emotion_set.get(field_key, emotion_set.get("default", "I hear you. Let's focus on taking things one step at a time."))
        
        return f"[Fallback Template] {fallback_msg}"

    # =======================================================
    # Subtask 3: Process Inputs and Keep Scores in Sync
    # =======================================================
    def execute_pipeline(self, field: str, problem_description: str) -> dict:
        # Run emotion prediction (ensures scores are dynamically calculated and fully in sync with current text)
        emotion_result = self.pipeline.predict(problem_description)
        
        # Generate custom guidance response
        guidance_text = self.generate_response(field, problem_description, emotion_result)
        
        # Build unified result payload
        result_state = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "field": field,
            "problem": problem_description,
            "emotion_result": emotion_result,
            "guidance_response": guidance_text
        }
        
        # Subtask 4: Log into session history & CSV files
        self.session_history.append(result_state)
        self.log_to_csv(result_state)
        
        return result_state

    # =======================================================
    # Subtask 4: CSV Logging Helper
    # =======================================================
    def log_to_csv(self, state: dict):
        file_exists = os.path.exists(self.log_path)
        
        headers = [
            "timestamp", "field", "problem", "primary_emotion", 
            "primary_confidence", "secondary_emotion", "secondary_confidence", 
            "is_mixed", "guidance_response"
        ]
        
        # Flattened row dictionary structure
        log_row = {
            "timestamp": state["timestamp"],
            "field": state["field"],
            "problem": state["problem"],
            "primary_emotion": state["emotion_result"]["primary_emotion"],
            "primary_confidence": state["emotion_result"]["primary_confidence"],
            "secondary_emotion": state["emotion_result"]["secondary_emotion"],
            "secondary_confidence": state["emotion_result"]["secondary_confidence"],
            "is_mixed": state["emotion_result"]["is_mixed"],
            "guidance_response": state["guidance_response"]
        }
        
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_row)

    def get_history(self):
        return self.session_history