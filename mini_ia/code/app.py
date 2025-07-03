from sklearn.linear_model import LogisticRegression
import joblib

X = [[2], [4], [6], [1], [3], [5]]
y = [0, 0, 0, 1, 1, 1]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model/parite.pkl")
