# Agregamos las librerias que vamos a requerir
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# =============== KNN clasificacion ============================
# Obtenemos los datos de entrada, en este caso el iris dataset https://archive.ics.uci.edu/ml/datasets/iris
X, y = load_iris(return_X_y=True)

# Instanciamos el modelo de KNN con K=5
neigh_model = KNeighborsClassifier(n_neighbors=5)

# Entrenamos el modelo
neigh_model.fit(X, y)

# Predecimos los primeros 3 datos
print(neigh_model.predict(X[:3, :]))

# Vemos la probabilidad de los primeros 3 datos
print(neigh_model.predict_proba(X[:3, :]))
