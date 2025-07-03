from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Charger le modèle entraîné
model = joblib.load("model/modele.pkl")

# Créer l'application FastAPI
app = FastAPI()

# Schéma des données attendues
class InputData(BaseModel):
    value: float

# Endpoint pour la prédiction
@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([[data.value]])
    return {"prediction": int(prediction[0])}
