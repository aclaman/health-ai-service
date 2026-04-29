from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model once when the service starts
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example phrases for each category
examples = {
    "low_concern": [
        "I feel well today.",
        "I slept badly but otherwise feel fine.",
        "I have mild tiredness but no other symptoms.",
        "I feel great."
    ],
    "needs_follow_up": [
        "I have been feeling dizzy for several days.",
        "I feel unusually tired and it is not improving.",
        "My symptoms are not severe but I am worried.",
        "I feel a little unwell",
        "I feel slightly sick."
    ],
    "urgent_review": [
        "I have chest pain and shortness of breath.",
        "I feel faint and have difficulty breathing.",
        "I have sudden weakness on one side of my body.",
        "I feel very sick."
    ]
}

# Pre-compute embeddings for all example phrases
category_embeddings = {
    category: model.encode(phrases)
    for category, phrases in examples.items()
}

# Define the input format
class TextInput(BaseModel):
    text: str

# Create the API
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(input: TextInput):
    # Embed the input text
    input_embedding = model.encode([input.text])[0]
    
    # Compare input to each category
    scores = {}
    for category, embeddings in category_embeddings.items():
        # Calculate similarity to each example in the category
        similarities = np.dot(embeddings, input_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_embedding)
        )
        # Take the highest similarity as the category score
        scores[category] = float(np.max(similarities))
    
    # Pick the category with the highest score
    best_label = max(scores, key=scores.get)
    confidence = scores[best_label]
    
    return {
        "label": best_label,
        "confidence": round(confidence, 2)
    }