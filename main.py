from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jellyfish
import re
import random
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util  # Para similaridade semântica
from typing import Dict, Any
from source.frases import frases_reprovado, frases_quase_la, frases_aprovado_pode_melhorar, frases_aprovado_bom_desempenho

app = FastAPI()

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

class EvaluationRequest(BaseModel):
    transcribed_text: str
    target_text: str

@app.post("/evaluate-pronunciation")
async def evaluate_pronunciation(request: EvaluationRequest) -> Dict[str, Any]:
    try:
        transcribed_clean = preprocess_text(request.transcribed_text)
        target_clean = preprocess_text(request.target_text)

        phonetic_score = calculate_phonetic_similarity(transcribed_clean, target_clean)

        semantic_score = calculate_semantic_similarity(transcribed_clean, target_clean)

        final_score = (phonetic_score * 0.85) + (semantic_score * 0.15)

        feedback = generate_feedback(final_score)

        return {
            "score": round(final_score * 100, 2),
            "feedback": feedback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_phonetic_similarity(text1: str, text2: str) -> float:
    words1 = text1.split()
    words2 = text2.split()

    if len(words1) != len(words2):
        return 0.0

    similarities = []
    for word1, word2 in zip(words1, words2):
        phonetic1 = jellyfish.metaphone(word1)
        phonetic2 = jellyfish.metaphone(word2)

        if phonetic1 == phonetic2:
            similarities.append(1.0)
        else:
            distance = jellyfish.levenshtein_distance(phonetic1, phonetic2)
            max_length = max(len(phonetic1), len(phonetic2))
            similarity = 1 - (distance / max_length) if max_length != 0 else 0
            similarities.append(similarity)

    return np.mean(similarities)

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    embeddings = semantic_model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity


def generate_feedback(score):
    if score < 0.6:
        return {"status": "Reprovado", "message": random.choice(frases_reprovado)}
    elif 0.6 <= score < 0.8:
        return {"status": "Quase lá", "message": random.choice(frases_quase_la)}
    elif 0.8 <= score < 0.95:
        return {"status": "Aprovado (Pode melhorar)", "message": random.choice(frases_aprovado_pode_melhorar)}
    else:
        return {"status": "Aprovado (Bom desempenho)", "message": random.choice(frases_aprovado_bom_desempenho)}


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()