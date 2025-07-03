
from sklearn.neighbors import KNeighborsClassifier
import joblib

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

prediction = model.predict([[1.5]])
print("Prédiction pour 1.5 :", prediction)

joblib.dump(model, "../model/modele.pkl")
