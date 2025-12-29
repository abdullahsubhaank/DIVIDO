from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import json
import numpy as np

app = FastAPI(title="Divido AI")

model = SentenceTransformer("all-MiniLM-L6-v2")
CATEGORY_FILE = "categories.json"


def load_categories():
    with open(CATEGORY_FILE, "r") as f:
        return json.load(f)


def save_categories(data):
    with open(CATEGORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class Email(BaseModel):
    text: str


class NewCategory(BaseModel):
    name: str
    examples: list[str]


@app.get("/")
def home():
    return {"message": "Divido AI is running"}


@app.post("/add-category")
def add_category(category: NewCategory):
    data = load_categories()
    data[category.name] = category.examples
    save_categories(data)
    return {"status": "success", "category": category.name}


@app.post("/predict")
def predict(email: Email):
    categories = load_categories()
    email_vec = model.encode(email.text)

    best_category = "Uncategorized"
    best_score = -1

    for category, examples in categories.items():
        scores = []
        for example in examples:
            example_vec = model.encode(example)
            scores.append(cosine_similarity(email_vec, example_vec))

        avg_score = sum(scores) / len(scores)

        if avg_score > best_score:
            best_score = avg_score
            best_category = category

    return {
        "app": "Divido",
        "category": best_category,
        "confidence": round(best_score, 2)
    }
