from guidance_engine import GuidanceEngine

def test_guidance_engine():
    # Initialize the unified Guidance Engine
    engine = GuidanceEngine()

    test_scenarios = [
        {
            "field": "Academic",
            "problem": "I am so overwhelmed by my final year project. I feel like I am going to fail and let everyone down."
        },
        {
            "field": "Career",
            "problem": "I just received an unexpected promotion today! I'm absolutely excited but slightly nervous about the sudden responsibilities."
        }
    ]

    print("\n--- Running AI-Powered Guidance & Regeneration Engine ---")
    for scenario in test_scenarios:
        print("\n" + "="*60)
        print(f"User Input Context: {scenario['field']}")
        print(f"User Problem Statement: \"{scenario['problem']}\"")
        
        # Execute pipeline (detects emotion, syncs scores, and requests Gemini response)
        result = engine.execute_pipeline(scenario["field"], scenario["problem"])
        
        emotion_data = result["emotion_result"]
        print(f"Primary Emotion:     {emotion_data['primary_emotion'].upper()} ({emotion_data['primary_confidence']*100:.2f}%)")
        if emotion_data['is_mixed']:
             print(f"Secondary Emotion:   {emotion_data['secondary_emotion'].upper()} ({emotion_data['secondary_confidence']*100:.2f}%)")
        
        print("\nGenerated AI Advisor Response:")
        print(result["guidance_response"])

    # Output verification check
    print("\n" + "="*60)
    print(f"Session History list count: {len(engine.get_history())} entry logged.")
    print("CSV file updated at: data/guidance_logs.csv")

if __name__ == "__main__":
    test_guidance_engine()