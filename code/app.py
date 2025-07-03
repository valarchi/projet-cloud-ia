from sklearn.neighbors import KNeighborsClassifier
import joblib

X = [[1.0], [2.0], [3.0], [4.0]]
y = [0, 0, 1, 1]

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

with open("data/input.txt", "r") as f:
    value = float(f.read().strip())

prediction = model.predict([[value]])

print(f"Prédiction pour {value} : {prediction}")



