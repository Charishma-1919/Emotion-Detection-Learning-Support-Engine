import os
import csv
import torch
import torch.nn as nn
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. BiLSTM Classifier Architecture Definition
# ==========================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(self.dropout(hidden_cat))
        return out


# ==========================================
# 2. Pipeline Manager (Cached Model Loading)
# ==========================================
class EmotionDetectionPipeline:
    _instance = None  # Singleton pattern to cache model loading in memory

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmotionDetectionPipeline, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Target 5 core classes
        self.emotions_5 = ["sadness", "joy", "love", "anger", "fear"]
        
        # Define Keyword Lexicon for Enhancement
        self.keyword_boosts = {
            "anger": ["angry", "mad", "hate", "furious", "annoyed", "pissed", "irritated"],
            "fear": ["scared", "afraid", "terrified", "anxious", "panic", "worried"],
            "joy": ["happy", "glad", "excited", "cheerful", "delighted", "awesome", "great"],
            "sadness": ["sad", "depressed", "crying", "gloomy", "sorrow", "hurt", "lonely"],
            "love": ["love", "adore", "affection", "romantic", "passionate", "sweet"]
        }
        
        # Static Class Weighting to balance minority/sensitive outputs
        self.class_weights = {
            "sadness": 1.0,
            "joy": 1.0,
            "love": 1.2,   # Slight boost for less frequent classes
            "anger": 1.1,
            "fear": 1.1
        }
        
        # Load local paths
        self.bert_path = os.path.join("models", "fine_tuned_bert")
        self.bilstm_path = os.path.join("models", "bilstm_emotion_model.pt")
        self.csv_path = os.path.join("data", "predictions.csv")
        
        # Ensure directories exist
        os.makedirs("data", exist_ok=True)
        
        print("Caching and Loading Models into Memory...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        
        # Load BERT Classifier
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(self.bert_path)
        self.bert_model.eval()
        
        # Load BiLSTM Classifier
        self.bilstm_model = BiLSTMClassifier(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=128,
            hidden_dim=128,
            num_classes=6  # Raw model was trained on 6 classes
        )
        self.bilstm_model.load_state_dict(torch.load(self.bilstm_path, map_location=torch.device('cpu')))
        self.bilstm_model.eval()
        
        self._initialized = True
        print("Models successfully loaded and cached!")

    # ==========================================
    # 3. Subtask 1: Text Preprocessing
    # ==========================================
    def preprocess_text(self, text: str) -> str:
        # Standardize text to lowercase and remove leading/trailing extra spacing
        cleaned = text.strip().lower()
        return cleaned

    # ==========================================
    # 4. Keyword Enhancement & Class Weighting
    # ==========================================
    def apply_pipeline_adjustments(self, raw_probs: dict, cleaned_text: str) -> dict:
        adjusted_probs = raw_probs.copy()
        
        # Apply Class Weighting Multipliers
        for emotion, weight in self.class_weights.items():
            adjusted_probs[emotion] *= weight
            
        # Apply Keyword Enhancement Boosts (+10% score boost per keyword match)
        for emotion, keywords in self.keyword_boosts.items():
            for kw in keywords:
                if kw in cleaned_text:
                    adjusted_probs[emotion] += 0.10
                    break  # Apply boost once per class matched
                    
        # Re-normalize values so they sum to 1.0 (Softmax-like scaling)
        total = sum(adjusted_probs.values())
        if total > 0:
            for emotion in adjusted_probs:
                adjusted_probs[emotion] /= total
                
        return adjusted_probs

    # ==========================================
    # 5. Core Pipeline Logic
    # ==========================================
    def predict(self, text: str) -> dict:
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize inputs
        inputs = self.tokenizer(cleaned_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        
        # A. BiLSTM Inference (Mapped to 5-Class Softmax)
        with torch.no_grad():
            bilstm_logits = self.bilstm_model(inputs["input_ids"])
            # Slice first 5 logits (to match sadness, joy, love, anger, fear) and apply Softmax
            bilstm_probs_tensor = torch.softmax(bilstm_logits[:, :5], dim=-1).flatten()
            bilstm_probs = {self.emotions_5[i]: bilstm_probs_tensor[i].item() for i in range(5)}
            
        # B. BERT Inference (Mapped to 5-Class Softmax with Class Weighting & Keywords)
        with torch.no_grad():
            bert_logits = self.bert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
            # Slice first 5 logits and apply Softmax
            bert_probs_tensor = torch.softmax(bert_logits[:, :5], dim=-1).flatten()
            bert_probs = {self.emotions_5[i]: bert_probs_tensor[i].item() for i in range(5)}
            
        bert_adjusted = self.apply_pipeline_adjustments(bert_probs, cleaned_text)
        
        # C. Subtask 5: Unified Prediction Schema (Ensemble: Weighted average of both models)
        # BERT is given 60% weight and BiLSTM is given 40% weight
        unified_scores = {}
        for emotion in self.emotions_5:
            unified_scores[emotion] = (0.6 * bert_adjusted[emotion]) + (0.4 * bilstm_probs[emotion])
            
        # Sort unified scores descending
        sorted_scores = sorted(unified_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Identify Primary Emotion
        primary_emotion, primary_conf = sorted_scores[0]
        
        # D. Subtask 4: Mixed-Emotion Detection (>= 15% Secondary Scores)
        secondary_emotion, secondary_conf = None, None
        is_mixed = False
        
        if len(sorted_scores) > 1:
            second_em, second_val = sorted_scores[1]
            if second_val >= 0.15:  # 15% threshold check
                secondary_emotion = second_em
                secondary_conf = second_val
                is_mixed = True
                
# Structured output schema
        prediction_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_text": text,
            "cleaned_text": cleaned_text,
            "primary_emotion": primary_emotion,
            "primary_confidence": round(primary_conf, 4),
            "secondary_emotion": secondary_emotion,
            "secondary_confidence": round(secondary_conf, 4) if secondary_conf else None,
            "is_mixed": is_mixed,
            
            # --- New lines added here ---
            "bilstm_scores": {k: round(v, 4) for k, v in bilstm_probs.items()},
            "bert_scores": {k: round(v, 4) for k, v in bert_adjusted.items()},
            # ----------------------------
            
            "all_scores": {k: round(v, 4) for k, v in unified_scores.items()}
        }
        
        # E. Subtask 6: CSV Persistence
        self.save_to_csv(prediction_result)
        
        return prediction_result

    def save_to_csv(self, result: dict):
        file_exists = os.path.exists(self.csv_path)
        
        headers = ["timestamp", "original_text", "primary_emotion", "primary_confidence", 
                   "secondary_emotion", "secondary_confidence", "is_mixed"]
        
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)