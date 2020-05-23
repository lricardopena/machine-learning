# Agregamos las librerias que vamos a requerir
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# Obtenemos los datos de entrada, en este caso el iris dataset https://archive.ics.uci.edu/ml/datasets/iris
X, y = load_iris(return_X_y=True)

# Instanciamos el modelo de perceptron
perceptron_model = Perceptron(tol=1e-3, random_state=0)

# Entrenamos el modelo
perceptron_model.fit(X, y)

# Predecimos los primeros 3 datos
print(perceptron_model.predict(X[:3, :]))

# Mostramos el score
print(perceptron_model.score(X, y))
