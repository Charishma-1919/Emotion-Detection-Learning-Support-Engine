# AI Learning Assistant: Emotion-Aware Student Support Pipeline

AI Learning Assistant is an educational support platform that analyzes student sentiment, detects underlying emotional states using a dual-inference NLP pipeline, and generates personalized, empathetic academic guidance using Gemini AI. 

The application is built on top of **Streamlit**, comparing a custom **BiLSTM (Bidirectional LSTM)** network, a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model, and **Google Gemini 1.5 Flash**.

---

## Features

- **Dual-Model Inference & Benchmarking:** Side-by-side comparison of local Recurrent Neural Networks (BiLSTM) and Transformer architectures (BERT) to analyze classification confidence and execution latency.
- **Robust Fallback Mechanism:** Automatically utilizes pre-trained public models (Hugging Face) if local Kaggle-trained checkpoints are not found, allowing immediate deployment out of the box.
- **Advanced Mixed-Emotion Detection:** Calculates the differential margin between the top two predicted emotions. If the margin is narrow ($\le 0.15$), the pipeline flags the state as a complex "mixed emotion" (e.g., combinations of fear and sadness).
- **Generative AI Counseling Guidance:** Dynamically constructs structured, 3-step intervention strategies utilizing Google Gemini AI.
- **Interactive Analytics Dashboard:** Real-time data visualization via Plotly (distribution plots, performance benchmarking, historical emotional trends over time).
- **Transaction Logging:** Auto-generates and appends student interactions, detected emotions, and model metrics to a persistent local `student_interactions.csv` database.

---

## System Architecture

```text
[ Student Input Text ]
          │
          ▼
[ NLTK Preprocessing Pipeline ]  ──► (Lowercasing, Stopword Removal, Lemmatization)
          │
          ├───► [ BiLSTM Network ] ───────► Class Probabilities
          │                                        │
          └───► [ BERT Transformer ] ─────► Class Probabilities
                                                   │
                                                   ▼
                                    [ Mixed-Emotion Decision Logic ]
                                                   │
                                                   ▼
                                     Final Assigned Emotion(s)
                                                   │
                                                   ├───► [ Plotly Charts & Dashboard ]
                                                   │
                                                   ▼
                                   [ Gemini AI Prompt Constructor ]
                                                   │
                                                   ▼
                                     Generated Intervention Plan
                                                   │
                                                   ▼
                                    [ CSV Logging Database (`.csv`) ]