import pytest
from fastapi.testclient import TestClient
from main import app  # Import du fichier principal

client = TestClient(app)

# === TESTS PREDICTION ===
def test_predict_positive():
    response = client.post("/predict", json={"text": "This is a great day!"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "positive"}

def test_predict_negative():
    response = client.post("/predict", json={"text": "This is a terrible day!"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "negative"}

def test_predict_empty():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Le texte d'entrée est vide"

# === TESTS FEEDBACK ===
def test_feedback_valid_positive():
    response = client.post("/feedback", json={
        "text": "This is a great day!",
        "prediction": "positive",
        "validation": True
    })
    assert response.status_code == 200
    assert response.json() == {"message": "Feedback enregistré, merci !"}

def test_feedback_invalid_prediction():
    response = client.post("/feedback", json={
        "text": "Some text",
        "prediction": "neutral",  # Erreur, "neutral" n'est pas une valeur acceptée
        "validation": False
    })
    assert response.status_code == 400
    assert "Prédiction invalide" in response.json()["detail"]

def test_feedback_misclassified():
    response = client.post("/feedback", json={
        "text": "This is a terrible day!",
        "prediction": "positive",  # Faux
        "validation": False
    })
    assert response.status_code == 200
    assert response.json() == {"message": "Feedback enregistré, merci !"}
