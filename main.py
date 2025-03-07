from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jellyfish
import re
import random
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from mangum import Mangum
import spacy

app = FastAPI()
handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_sm")

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
    # Usa spaCy para gerar embeddings e calcular similaridade
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity = doc1.similarity(doc2)  # Similaridade cosseno entre os embeddings
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

# FRASES

frases_reprovado = [
    "Tente de novo! Você está no caminho certo!",
    "Quase lá! Vamos tentar mais uma vez?",
    "Não desista! Tente de novo e você vai conseguir!",
    "Está difícil? Respire fundo e tente mais uma vez!",
    "Vamos tentar mais uma vez? Você pode melhorar!",
    "Isso foi um bom começo! Agora tente mais uma vez com atenção.",
    "Errar faz parte do aprendizado! Vamos tentar de novo?",
    "Você está aprendendo! Mais uma tentativa e vai sair melhor!",
]

frases_quase_la = [
    "Está quase lá! Experimente mais uma vez!",
    "Muito bom! Falta só um pouquinho para ficar perfeito!",
    "Você já melhorou bastante! Mais uma tentativa e você acerta!",
    "Bem perto! Se quiser, tente de novo para acertar tudo!",
    "Ótimo esforço! Só mais um pouquinho e você chega lá!",
    "Muito bem! Tente mais uma vez para ficar ainda melhor!",
    "Você está quase acertando! Vamos mais uma tentativa?",
    "Sua pronúncia está muito boa! Só mais um ajuste!",
]

frases_aprovado_pode_melhorar = [
    "Muito bom! Se quiser, pode tentar mais uma vez para ficar ainda melhor!",
    "Ótima tentativa! Já está bom, mas pode ficar excelente!",
    "Parabéns! Se quiser, tente mais uma vez para aperfeiçoar!",
    "Bom trabalho! Está ótimo, mas que tal um desafio e tentar de novo?",
    "Mandou muito bem! Mas se quiser um desafio, tente mais uma vez!",
    "Você acertou! Mas se quiser melhorar ainda mais, pode tentar de novo!",
    "Muito bom! Quer tentar de novo para ver se consegue um resultado ainda melhor?",
    "Já está ótimo! Mas se quiser, tente mais uma vez para deixar perfeito!",
]

frases_aprovado_bom_desempenho = [
    "Incrível! Você mandou muito bem!",
    "Perfeito! Continue assim!",
    "Ótimo trabalho! Sua pronúncia está excelente!",
    "Muito bem! Você está falando como um nativo!",
    "Fantástico! Continue praticando desse jeito!",
    "Uau! Você acertou em cheio!",
    "Que incrível! Continue assim!",
    "Parabéns! Você fez um ótimo trabalho!",
]