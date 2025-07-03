from fastapi import FastAPI
import joblib


model = joblib.load("model/modele.pkl")

app = FastAPI()
@app.get("/")
def accueil():
    return {"message": "Bienvenue sur l'API IA de prédiction. Utilisez /predict pour envoyer une valeur."}


@app.post("/predict")
def predict(valeur: dict):
    prediction = model.predict([[valeur["valeur"]]])
    return {"prediction": prediction[0]}

