import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = None
model = None


def load_mentalbert():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("mentalbert_model")
    model = AutoModelForSequenceClassification.from_pretrained("mentalbert_model")
    model.eval()

    return tokenizer, model


def predict_mental_state(text):
    global tokenizer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()

    labels = ["anxiety", "depression", "mental_disorder", "normal", "suicidewatch"]

    return labels[pred], confidence
