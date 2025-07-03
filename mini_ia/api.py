from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("model/parite.pkl")

app = FastAPI()

class InputData(BaseModel):
    value: int

@app.post("/check")
def check(data: InputData):
    prediction = model.predict([[data.value]])
    label = "Impair" if prediction[0] == 1 else "Pair"
    return {"valeur": data.value, "parité": label}
