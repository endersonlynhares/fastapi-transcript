from fastapi import FastAPI, HTTPException, Body
import jellyfish
import re
import random
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from mangum import Mangum
import spacy
import base64
import json

app = FastAPI()
handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_md")

@app.post("/track-evlp")
async def evaluate_pronunciation(encoded_data: str = Body(...)) -> Dict[str, Any]:
    try:
        json_data = decode_with_protection(encoded_data)
        print(json_data)
        transcribed_text = json_data.get("transcribed_text", "")
        target_text = json_data.get("target_text", "")

        transcribed_clean = preprocess_text(transcribed_text)
        target_clean = preprocess_text(target_text)

        transcribed_words = transcribed_clean.split()
        target_words = target_clean.split()

        correct_words = []
        incorrect_words = []
        phonetic_details = []
        correct_count = 0
        incorrect_count = 0

        for transcribed_word, target_word in zip(transcribed_words, target_words):
            phonetic_transcribed = jellyfish.metaphone(transcribed_word)
            phonetic_target = jellyfish.metaphone(target_word)

            if phonetic_transcribed == phonetic_target:
                correct_words.append(transcribed_word)
                correct_count += 1
            else:
                incorrect_words.append(transcribed_word)
                incorrect_count += 1

            phonetic_details.append({
                "transcribed_word": transcribed_word,
                "target_word": target_word,
                "phonetic_transcribed": phonetic_transcribed,
                "phonetic_target": phonetic_target,
                "is_correct": phonetic_transcribed == phonetic_target
            })

        phonetic_score = calculate_phonetic_similarity(transcribed_clean, target_clean)
        semantic_score = calculate_semantic_similarity(transcribed_clean, target_clean)
        final_score = (phonetic_score * 0.90) + (semantic_score * 0.10)
        feedback = generate_feedback(final_score)

        return {
            "score": round(final_score * 100, 2),
            "feedback": feedback,
            "correct_words": correct_words,
            "incorrect_words": incorrect_words,
            "phonetic_details": phonetic_details,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count
        }

    except Exception as e:
        print(e)
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
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity = doc1.similarity(doc2)
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

def decode_with_protection(protected_base64: str) -> dict:
    clean_base64 = (
        protected_base64[:2] +  # Remove o caractere na 3ª posição
        protected_base64[3:7] +  # Remove o caractere na 7ª posição
        protected_base64[8:-1]  # Remove o caractere na última posição
    )

    json_str = base64.b64decode(clean_base64).decode('utf-8')
    print(clean_base64)
    print(json_str)
    print(json.loads(json_str))
    return json.loads(json_str)

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